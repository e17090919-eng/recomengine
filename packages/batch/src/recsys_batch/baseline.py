"""Offline baseline report (P0 deliverable).

Time split: train on everything before `end - test_days`, evaluate on the last
`test_days`. Works on real customer data; if an `oracle` (ground-truth click model,
available only for synthetic data) is supplied, also reports simulated CTR.

Reported:
  * ranking quality on test impressions: AUC / GAUC of the GBDT vs single-signal
    baselines (hot score, Swing affinity, embedding similarity)
  * top-K quality of full strategies: HitRate / Recall / NDCG @K, coverage, diversity
  * per-channel recall, latency
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from rec_server.data import DAY, Tables
from rec_server.engine import Engine, Request
from rec_server.features import FEATURES
from rec_server.pipeline import build_engine, build_samples
from rec_server.rank import gauc

Oracle = Callable[[str, np.ndarray], np.ndarray]  # (user_id, item_ids) -> P(click)

STRATEGIES: dict[str, dict] = {
    "Hot": dict(recalls=[{"name": "hot", "quota": 300, "weight": 1.0}], ranker="recall_score"),
    "Swing": dict(recalls=[{"name": "swing", "quota": 300, "weight": 1.0}], ranker="recall_score"),
    "Embedding": dict(recalls=[{"name": "embedding", "quota": 300, "weight": 1.0}], ranker="recall_score"),
    "Multi-recall (fusion)": dict(ranker="recall_score"),
    "Multi-recall + GBDT": dict(ranker="gbdt"),
    "Multi-recall + GBDT, no diversity rules": dict(ranker="gbdt", dispersion=[], boost_new=None),
}
POSITIVE_EVENTS = ("click", "add_cart", "purchase")


@dataclass
class TopK:
    hit: float = 0.0
    recall: float = 0.0
    ndcg: float = 0.0
    n: int = 0

    def add(self, rec: list[int], pos: set[int], k: int) -> None:
        hits = [1.0 if i in pos else 0.0 for i in rec[:k]]
        self.hit += float(any(hits))
        self.recall += sum(hits) / len(pos)
        idcg = sum(1 / np.log2(r + 2) for r in range(min(len(pos), k)))
        self.ndcg += sum(h / np.log2(r + 2) for r, h in enumerate(hits)) / idcg
        self.n += 1

    def row(self) -> tuple[float, float, float]:
        n = max(self.n, 1)
        return self.hit / n, self.recall / n, self.ndcg / n


def ranking_metrics(
    engine: Engine, t: Tables, config: dict, start: float, end: float
) -> list[tuple[str, float, float]]:
    """AUC / GAUC on test-window impressions with point-in-time features."""
    X, y, g, _ = build_samples(t, engine.catalog, config, start, end)
    if y.sum() == 0 or y.sum() == len(y):
        return []
    rows = []
    scorers = {
        "GBDT ranker": engine.ranker.predict(X) if engine.ranker is not None else None,
        "Hot score (market)": X[:, FEATURES.index("item_market_hot")],
        "Swing affinity": X[:, FEATURES.index("swing_aff")],
        "Embedding similarity": X[:, FEATURES.index("emb_dot")],
        "Item CTR (smoothed)": X[:, FEATURES.index("item_ctr")],
    }
    for name, s in scorers.items():
        if s is not None:
            rows.append((name, float(roc_auc_score(y, s)), float(gauc(y, s, g))))
    return rows


def run(
    config: dict,
    lake_dir: str,
    test_days: int = 5,
    n_users: int = 1000,
    k: int = 20,
    seed: int = 0,
    oracle: Oracle | None = None,
    log: Callable[..., None] = print,
) -> str:
    t = Tables.load(lake_dir)
    end = t.end_ts
    cutoff = end - test_days * DAY
    engine = build_engine(config, t, cutoff, log=log)
    log("[baseline] ranking metrics on test impressions")
    rank_rows = ranking_metrics(engine, t, config, cutoff, end)

    b = t.behaviors
    test = b[(b["ts"] >= cutoff) & b["event_type"].isin(POSITIVE_EVENTS)]
    test = test[test["item_id"].isin(engine.catalog.id2idx)]
    pos = test.groupby("user_id")["item_id"].apply(lambda s: {engine.catalog.id2idx[x] for x in s})
    rng = np.random.default_rng(seed)
    users = rng.choice(pos.index.to_numpy(), size=min(n_users, len(pos)), replace=False)
    known = {u for u in users if engine.users.history(u)}
    slot = "home_feed" if "home_feed" in config["slots"] else next(iter(config["slots"]))
    spm = config["slots"][slot]["spm"] + "$##$0"

    log(f"[baseline] top-{k} on {len(users)} users")
    strat_rows: list[dict[str, Any]] = []
    for name, ov in STRATEGIES.items():
        all_k, known_k = TopK(), TopK()
        ctr_all, ctr_known, cats, lat = [], [], [], []
        shown: set[int] = set()
        for u in users:
            rec = engine.recommend(Request(user_id=u, spm=spm, size=k), overrides=ov, record=False)
            ix = [it["idx"] for it in rec.items]
            all_k.add(ix, pos[u], k)
            shown |= set(ix)
            cats.append(len({engine.catalog.cat[i] for i in ix}))
            lat.append(rec.trace["total_ms"])
            c = float(np.nanmean(oracle(u, engine.catalog.item_ids[ix]))) if oracle is not None and ix else np.nan
            ctr_all.append(c)
            if u in known:
                known_k.add(ix, pos[u], k)
                ctr_known.append(c)
        strat_rows.append(
            dict(
                name=name,
                all=all_k.row(),
                known=known_k.row(),
                cov=len(shown) / engine.catalog.n,
                cats=float(np.mean(cats)),
                p50=float(np.percentile(lat, 50)),
                p99=float(np.percentile(lat, 99)),
                ctr=float(np.nanmean(ctr_all)) if oracle else None,
                ctr_known=float(np.nanmean(ctr_known)) if oracle and ctr_known else None,
            )
        )

    chan_rows = []
    for rc in config["slots"][slot]["recalls"]:
        got = 0.0
        for u in users:
            rec = engine.recommend(
                Request(user_id=u, spm=spm, size=rc["quota"]),
                overrides=dict(recalls=[rc], ranker="recall_score", dispersion=[]),
                record=False,
            )
            got += len({it["idx"] for it in rec.items} & pos[u]) / len(pos[u])
        chan_rows.append((rc["name"], rc["quota"], got / len(users)))

    m = engine.info["ranker"]
    fmt = lambda ts: f"{np.datetime64(int(ts), 's')}"[:10]  # noqa: E731
    L = [
        "# Offline baseline report",
        "",
        f"- Data: `{lake_dir}` ({t.meta.get('source', 'lake')}), {len(t.behaviors):,} events, "
        f"{len(t.users):,} users, {len(t.items):,} items, markets: {', '.join(engine.catalog.markets)}",
        f"- Split: train < {fmt(cutoff)} ≤ test < {fmt(end)} ({test_days} days); "
        f"{len(users)} test users with future engagement, {len(known)} of them with history (others are cold start)",
        f"- Ranker: {m['backend']}, trained on {m['train_rows']:,} impressions; validation AUC {m['auc']:.4f}, "
        f"GAUC {m['gauc']:.4f}",
        "",
    ]
    if rank_rows:
        L += [
            "## 1. Ranking quality on test impressions",
            "",
            "How well each signal orders what users actually saw (click = positive). GAUC = per-user AUC.",
            "",
            "| scorer | AUC | GAUC |",
            "|---|---|---|",
        ]
        L += [f"| {n} | {a:.4f} | {g:.4f} |" for n, a, g in rank_rows]
        L.append("")
    L += [
        f"## 2. Top-{k} recommendation quality",
        "",
        "Positives = items the user clicked / carted / bought in the test window.",
        "",
    ]
    head = f"| strategy | HitRate@{k} | Recall@{k} | NDCG@{k} | HitRate@{k} (w/ history) | NDCG@{k} (w/ history) |"
    sep = "|---|---|---|---|---|---|"
    if oracle:
        head += " sim. CTR | sim. CTR (w/ history) |"
        sep += "---|---|"
    head += " item coverage | categories / list | P50 ms | P99 ms |"
    sep += "---|---|---|---|"
    L += [head, sep]
    base = strat_rows[0]
    for r in strat_rows:
        line = (
            f"| {r['name']} | {r['all'][0]:.3f} | {r['all'][1]:.3f} | {r['all'][2]:.4f} | "
            f"{r['known'][0]:.3f} | {r['known'][2]:.4f} |"
        )
        if oracle:
            line += (
                f" {r['ctr']:.4f} ({r['ctr'] / base['ctr'] - 1:+.1%}) | "
                f"{r['ctr_known']:.4f} ({r['ctr_known'] / base['ctr_known'] - 1:+.1%}) |"
            )
        line += f" {r['cov']:.1%} | {r['cats']:.1f} | {r['p50']:.1f} | {r['p99']:.1f} |"
        L.append(line)
    L += [
        "",
        "## 3. Recall channels",
        "",
        "Share of each user's future positives found within the channel's quota.",
        "",
        "| channel | quota | recall |",
        "|---|---|---|",
    ]
    L += [f"| {c} | {q} | {v:.3f} |" for c, q, v in chan_rows]
    L += [
        "",
        "## Notes",
        "",
        "- HitRate/Recall/NDCG use *logged* engagement, which was shaped by the incumbent system's exposure "
        "policy (usually popularity-heavy). That biases these metrics toward `Hot`; online A/B is the final "
        "judge.",
    ]
    if oracle:
        L.append(
            "- *Simulated CTR* uses the synthetic data's hidden preference model as an unbiased stand-in for "
            "an online test (lift in brackets vs `Hot`)."
        )
    L.append("- Cold-start users (no history before the split) receive Hot / new-item results in every strategy.")
    return "\n".join(L) + "\n"

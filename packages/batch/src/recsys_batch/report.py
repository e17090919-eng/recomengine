"""T+1 daily report: attribution, funnel by slot / market / variant, and A/B readout.

Attribution
  - impressions / clicks carry the `request_id` of the recommendation that showed them;
  - add_cart / order_submit / purchase with a request_id are attributed directly, otherwise
    to the user's last click on the same item within `window_h` hours (last-click);
  - slot and variant come from the engine's request log (`rec_log`); for data without a
    request log (historical backfill) the slot is routed from the SPM and the variant is
    re-derived from the deterministic A/B hash (marked as inferred).

A/B readout (per slot x market, every variant vs the first one = control)
  - CTR as a ratio metric with a user-level delta-method CI, GMV / exposed user with a
    Welch CI, 95% two-sided; plus a sample-ratio-mismatch (SRM) chi-square check.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from recsys_core.lake import Lake

CONV_EVENTS = ("add_cart", "order_submit", "purchase")
Z = 1.959963984540054


def _norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2))


def _chi2_sf(x: float, k: int) -> float:
    """Survival function of chi-square with k dof (Wilson–Hilferty approximation; fine for SRM alerts)."""
    if k <= 0:
        return 1.0
    z = ((x / k) ** (1 / 3) - (1 - 2 / (9 * k))) / math.sqrt(2 / (9 * k))
    return _norm_sf(z)


@dataclass
class Report:
    date: str
    rows: pd.DataFrame  # slot x market x variant metrics
    ab: list[dict[str, Any]] = field(default_factory=list)
    srm: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "date": self.date,
                "meta": self.meta,
                "rows": self.rows.to_dict("records"),
                "ab": self.ab,
                "srm": self.srm,
            },
            ensure_ascii=False,
            indent=2,
            default=float,
        )

    def to_markdown(self) -> str:
        m = self.meta
        out = [
            f"# Daily recommendation report — {self.date}",
            "",
            f"Business date per market time zone. Events in window: **{m.get('events', 0):,}** · "
            f"requests logged: **{m.get('requests', 0):,}** · variant source: **{m.get('variant_source')}** · "
            f"attribution: last click ≤ {m.get('window_h')}h",
            "",
            "## Funnel by slot · market · variant",
            "",
            "| slot | market | variant | requests | exposed users | impressions | clicks | CTR | carts | orders "
            "| purchases | GMV | GMV / exposed user | degraded |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in self.rows.to_dict("records"):
            out.append(
                f"| {r['slot']} | {r['market']} | {r['variant']} | {r['requests']:,} | {r['exposed_users']:,} "
                f"| {r['impressions']:,} | {r['clicks']:,} | {r['ctr']:.2%} | {r['add_cart']:,} "
                f"| {r['order_submit']:,} "
                f"| {r['purchase']:,} | {r['gmv']:,.0f} {r['currency']} | {r['gmv_per_user']:.2f} "
                f"| {r['degraded_rate']:.2%} |"
            )
        organic = m.get("organic_conversions", {})
        if organic:
            out += ["", "Conversions not attributable to a recommendation (organic): " + ", ".join(
                f"{k}={v:,}" for k, v in organic.items()
            )]  # fmt: skip
        out += ["", "## A/B readout (95% CI, vs control)", ""]
        if not self.ab:
            out.append("_No experiment with ≥ 2 variants in this window._")
        else:
            out += [
                "| slot | market | layer | variant | metric | control | variant value | lift | 95% CI | p | |",
                "|---|---|---|---|---|---:|---:|---:|---|---:|---|",
            ]
            for a in self.ab:
                sig = "✅" if a["p"] < 0.05 and a["lift"] > 0 else ("❌" if a["p"] < 0.05 else "—")
                fmt = (lambda v: f"{v:.2%}") if a["metric"] == "CTR" else (lambda v: f"{v:.2f}")
                out.append(
                    f"| {a['slot']} | {a['market']} | {a['layer']} | {a['variant']} | {a['metric']} "
                    f"| {fmt(a['control'])} | {fmt(a['value'])} | {a['lift']:+.1%} "
                    f"| [{a['ci_low']:+.1%}, {a['ci_high']:+.1%}] | {a['p']:.3g} | {sig} |"
                )
        if self.srm:
            out += ["", "## Sample ratio check", ""]
            for s in self.srm:
                flag = "⚠️ SRM" if s["p"] < 0.001 else "ok"
                out.append(
                    f"- {s['slot']} / {s['layer']}: users {s['observed']} vs expected share {s['expected']} "
                    f"(p={s['p']:.3g}) {flag}"
                )
        return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- loading
def _local_window(date: str, tz: str) -> tuple[int, int]:
    d = datetime.strptime(date, "%Y%m%d").replace(tzinfo=ZoneInfo(tz))
    start = int(d.timestamp())
    return start, start + 86400  # DST days are off by an hour at most; acceptable for a daily report


def _route(spm: pd.Series, slots: dict[str, dict]) -> pd.Series:
    prefix = spm.fillna("").astype(str).str.split("$##$", regex=False).str[:3].str.join("$##$")
    lookup = {s["spm"]: name for name, s in slots.items()}
    return prefix.map(lookup)


def _infer_variants(users: pd.Series, slot: pd.Series, cfg: dict) -> pd.Series:
    from rec_server.abtest import assign

    exps = cfg.get("experiments", {})
    cache: dict[tuple[str, str], str] = {}

    def one(u: str, s: str) -> str:
        k = (u, s)
        if k not in cache:
            cache[k] = ",".join(f"{a.layer}={a.variant}" for a in assign(exps, u, s))
        return cache[k]

    return pd.Series(
        [one(str(u), str(s)) if isinstance(s, str) else "" for u, s in zip(users, slot)], index=users.index
    )


def load_day(lake: Lake, cfg: dict, date: str, window_h: float = 72) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Behaviors of the local business `date` in every market (+ the look-back needed for
    last-click attribution) and the matching request log."""
    windows = {m: _local_window(date, c.get("timezone", "UTC")) for m, c in cfg["markets"].items()}
    lo = min(w[0] for w in windows.values()) - int(window_h * 3600)
    hi = max(w[1] for w in windows.values())
    b = lake.read("behavior", lo, hi)
    log = lake.read("rec_log", lo, hi)
    start = b["market"].map({m: w[0] for m, w in windows.items()})
    end = b["market"].map({m: w[1] for m, w in windows.items()})
    b["in_day"] = (b["ts"] >= start) & (b["ts"] < end)
    return b, log, windows


# --------------------------------------------------------------------------- attribution
def attribute(b: pd.DataFrame, log: pd.DataFrame, cfg: dict, window_h: float = 72) -> tuple[pd.DataFrame, str]:
    """Adds slot / variants / attributed request_id to every behavior row."""
    b = b.copy()
    b["request_id"] = b["request_id"].fillna("0").astype(str)
    has_req = b["request_id"] != "0"
    b["attr_rid"] = b["request_id"].where(has_req, None)

    # last-click attribution for conversions without a request id
    clicks = b[(b["event_type"] == "click") & has_req][["user_id", "item_id", "ts", "request_id"]].sort_values("ts")
    conv = b[b["event_type"].isin(CONV_EVENTS) & ~has_req].reset_index()
    if len(conv) and len(clicks):
        m = pd.merge_asof(
            conv.sort_values("ts"),
            clicks.rename(columns={"request_id": "click_rid", "ts": "click_ts"}).sort_values("click_ts"),
            left_on="ts",
            right_on="click_ts",
            by=["user_id", "item_id"],
            direction="backward",
            tolerance=int(window_h * 3600),
        )
        b.loc[m["index"].to_numpy(), "attr_rid"] = m["click_rid"].to_numpy()

    if len(log):
        rl = log.drop_duplicates("request_id").set_index("request_id")
        b["slot"] = b["attr_rid"].map(rl["slot"])
        b["variants"] = b["attr_rid"].map(rl["variants"]).fillna("")
        b["degraded"] = b["attr_rid"].map(rl["degraded"]).fillna("")
        source = "request log"
        # rows whose request is not in the log (e.g. log retention) fall back to routing
        miss = b["attr_rid"].notna() & b["slot"].isna()
    else:
        b["slot"], b["variants"], b["degraded"] = None, "", ""
        source = "inferred (hash)"
        miss = b["attr_rid"].notna()
    if miss.any():
        rid_spm = b[b["attr_rid"].notna() & has_req].groupby("request_id")["spm"].first()
        spm = b.loc[miss, "attr_rid"].map(rid_spm)
        b.loc[miss, "slot"] = _route(spm, cfg["slots"]).to_numpy()
        b.loc[miss, "variants"] = _infer_variants(b.loc[miss, "user_id"], b.loc[miss, "slot"], cfg).to_numpy()
        if len(log):
            source = "request log + inferred"
    return b, source


# --------------------------------------------------------------------------- metrics
def _split_variant(v: str, layer: str) -> str:
    for kv in str(v or "").split(","):
        if kv.startswith(layer + "="):
            return kv.split("=", 1)[1]
    return "-"


def daily_report(lake_dir: str, cfg: dict, date: str, window_h: float = 72) -> Report:
    lake = Lake(lake_dir)
    b, log, windows = load_day(lake, cfg, date, window_h)
    b, source = attribute(b, log, cfg, window_h)
    day = b[b["in_day"]].copy()
    rec = day[day["slot"].notna()].copy()
    layers = list(cfg.get("experiments", {}))
    rec["variant"] = rec["variants"].map(lambda v: "|".join(_split_variant(v, la) for la in layers) or "-")
    currency = {m: c["currency"] for m, c in cfg["markets"].items()}

    if len(log):
        lg = log.copy()
        ls = lg["market"].map({m: w[0] for m, w in windows.items()})
        le = lg["market"].map({m: w[1] for m, w in windows.items()})
        lg = lg[(lg["ts"] >= ls) & (lg["ts"] < le)]
        lg["variant"] = lg["variants"].map(lambda v: "|".join(_split_variant(v, la) for la in layers) or "-")
        req = lg.groupby(["slot", "market", "variant"]).agg(
            requests=("request_id", "nunique"), degraded=("degraded", lambda s: (s.fillna("") != "").mean())
        )
    else:
        req = pd.DataFrame(columns=["requests", "degraded"])

    rows = []
    for (slot, market, variant), g in rec.groupby(["slot", "market", "variant"]):
        cnt = g["event_type"].value_counts()
        exposed = g.loc[g["event_type"] == "impression", "user_id"].nunique()
        gmv = float(g.loc[g["event_type"] == "purchase", "paid_amount"].fillna(0).sum())
        imp = int(cnt.get("impression", 0))
        key: Any = (slot, market, variant)
        r = req.loc[key] if key in req.index else None
        rows.append(
            dict(
                slot=slot,
                market=market,
                variant=variant,
                requests=int(r["requests"]) if r is not None else int(g["attr_rid"].nunique()),
                exposed_users=int(exposed),
                impressions=imp,
                clicks=int(cnt.get("click", 0)),
                ctr=float(cnt.get("click", 0) / imp) if imp else 0.0,
                add_cart=int(cnt.get("add_cart", 0)),
                order_submit=int(cnt.get("order_submit", 0)),
                purchase=int(cnt.get("purchase", 0)),
                gmv=gmv,
                currency=currency.get(market, ""),
                gmv_per_user=gmv / exposed if exposed else 0.0,
                degraded_rate=float(r["degraded"]) if r is not None else 0.0,
            )
        )
    table = pd.DataFrame(rows)
    organic = day[day["slot"].isna() & day["event_type"].isin(CONV_EVENTS)]["event_type"].value_counts()
    meta = dict(
        events=len(day),
        requests=len(log),
        variant_source=source,
        window_h=window_h,
        organic_conversions={k: int(v) for k, v in organic.items()},
        windows={m: [int(a), int(z)] for m, (a, z) in windows.items()},
    )
    rep = Report(date, table, meta=meta)
    rep.ab, rep.srm = ab_readout(rec, cfg)
    return rep


def _user_level(g: pd.DataFrame) -> pd.DataFrame:
    et = g["event_type"]
    u = pd.DataFrame(
        {
            "imp": (et == "impression").groupby(g["user_id"]).sum(),
            "clk": (et == "click").groupby(g["user_id"]).sum(),
            "gmv": g["paid_amount"].where(et == "purchase", 0).fillna(0).groupby(g["user_id"]).sum(),
        }
    )
    return u[u["imp"] > 0]


def _ratio_lift(c: pd.DataFrame, t: pd.DataFrame) -> tuple[float, float, float, float, float]:
    """CTR = sum(clk)/sum(imp); lift of t over c with delta-method variance (user = unit)."""

    def est(u: pd.DataFrame) -> tuple[float, float]:
        n = len(u)
        mx, my = u["imp"].mean(), u["clk"].mean()
        r = my / mx
        vx, vy = u["imp"].var(ddof=1), u["clk"].var(ddof=1)
        cxy = np.cov(u["imp"], u["clk"], ddof=1)[0, 1]
        var = (vy - 2 * r * cxy + r * r * vx) / (mx * mx) / n
        return r, var

    rc, vc = est(c)
    rt, vt = est(t)
    lift = rt / rc - 1
    se = math.sqrt(vt / rc**2 + (rt**2) * vc / rc**4)  # delta method for rt/rc
    p = 2 * _norm_sf(abs(lift) / se) if se > 0 else 1.0
    return rc, rt, lift, se, p


def _mean_lift(c: pd.Series, t: pd.Series) -> tuple[float, float, float, float, float]:
    mc, mt = c.mean(), t.mean()
    if mc == 0:
        return mc, mt, 0.0, float("inf"), 1.0
    lift = mt / mc - 1
    se = math.sqrt(t.var(ddof=1) / len(t) / mc**2 + mt**2 * c.var(ddof=1) / len(c) / mc**4)
    p = 2 * _norm_sf(abs(lift) / se) if se > 0 else 1.0
    return mc, mt, lift, se, p


def ab_readout(rec: pd.DataFrame, cfg: dict, min_users: int = 30) -> tuple[list[dict], list[dict]]:
    ab, srm = [], []
    for layer, exp in cfg.get("experiments", {}).items():
        names = [v["name"] for v in exp["variants"]]
        traffic = {v["name"]: v["traffic"] for v in exp["variants"]}
        if len(names) < 2:
            continue
        for slot in exp["slots"]:
            s = rec[rec["slot"] == slot].copy()
            if s.empty:
                continue
            s["v"] = s["variants"].map(lambda x, la=layer: _split_variant(x, la))
            users = s[s["event_type"] == "impression"].groupby("v")["user_id"].nunique()
            obs = [int(users.get(n, 0)) for n in names]
            if sum(obs):
                tot, share = sum(obs), sum(traffic.values())
                exp_n = [tot * traffic[n] / share for n in names]
                x2 = sum((o - e) ** 2 / e for o, e in zip(obs, exp_n) if e > 0)
                srm.append(
                    dict(
                        slot=slot,
                        layer=layer,
                        observed=dict(zip(names, obs)),
                        expected={n: traffic[n] for n in names},
                        p=_chi2_sf(x2, len(names) - 1),
                    )
                )
            for market, sm in s.groupby("market"):
                per = {n: _user_level(sm[sm["v"] == n]) for n in names}
                ctrl = per[names[0]]
                if len(ctrl) < min_users:
                    continue
                for n in names[1:]:
                    t = per[n]
                    if len(t) < min_users:
                        continue
                    for metric, res in (
                        ("CTR", _ratio_lift(ctrl, t)),
                        ("GMV / exposed user", _mean_lift(ctrl["gmv"], t["gmv"])),
                    ):
                        c0, v1, lift, se, p = res
                        ab.append(
                            dict(
                                slot=slot,
                                market=market,
                                layer=layer,
                                variant=n,
                                metric=metric,
                                control=float(c0),
                                value=float(v1),
                                lift=float(lift),
                                ci_low=float(lift - Z * se),
                                ci_high=float(lift + Z * se),
                                p=float(p),
                                users_control=len(ctrl),
                                users_variant=len(t),
                            )
                        )
    return ab, srm

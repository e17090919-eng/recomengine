"""Synthetic travel-retail data with a hidden ground-truth preference model.

Produces the same three tables a real customer would send (users / items /
behaviors, plus per-market item availability), and saves the latent truth so
offline evaluation can estimate "online" CTR of any strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DAY = 86400
START_TS = 1790812800  # 2026-10-01 00:00:00 UTC
DIM = 16

# name, zh name, placeholder image, (min, max) USD price
CATEGORIES = [
    ("skincare", "护肤", "🧴", (25, 400)),
    ("makeup", "彩妆", "💄", (15, 120)),
    ("fragrance", "香水", "🌸", (50, 350)),
    ("liquor", "烈酒", "🥃", (30, 1500)),
    ("wine", "葡萄酒", "🍷", (20, 600)),
    ("watches", "腕表", "⌚", (200, 15000)),
    ("jewelry", "珠宝", "💍", (100, 6000)),
    ("handbags", "箱包", "👜", (200, 5000)),
    ("confectionery", "糖果零食", "🍫", (5, 60)),
    ("health", "保健品", "🍯", (15, 200)),
    ("electronics", "数码", "🎧", (40, 1200)),
    ("eyewear", "眼镜", "🕶️", (80, 600)),
]
NOUNS = {
    "skincare": [
        ("Serum", "精华"),
        ("Cream", "面霜"),
        ("Essence", "精华水"),
        ("Eye Cream", "眼霜"),
        ("Mask Set", "面膜套装"),
    ],
    "makeup": [
        ("Lipstick", "口红"),
        ("Foundation", "粉底液"),
        ("Palette", "眼影盘"),
        ("Cushion", "气垫"),
        ("Mascara", "睫毛膏"),
    ],
    "fragrance": [
        ("Eau de Parfum", "香水"),
        ("Eau de Toilette", "淡香水"),
        ("Gift Set", "香氛礼盒"),
        ("Body Mist", "香体喷雾"),
    ],
    "liquor": [
        ("Single Malt 12Y", "单一麦芽12年"),
        ("Cognac XO", "干邑XO"),
        ("Baijiu", "白酒"),
        ("Gin", "金酒"),
        ("Whisky 18Y", "威士忌18年"),
    ],
    "wine": [("Pinot Noir", "黑皮诺"), ("Sauvignon Blanc", "长相思"), ("Champagne", "香槟"), ("Bordeaux", "波尔多")],
    "watches": [
        ("Automatic", "自动机械表"),
        ("Chronograph", "计时腕表"),
        ("Diver", "潜水表"),
        ("Dress Watch", "正装表"),
    ],
    "jewelry": [("Necklace", "项链"), ("Bracelet", "手链"), ("Ring", "戒指"), ("Earrings", "耳饰")],
    "handbags": [("Tote", "托特包"), ("Crossbody", "斜挎包"), ("Wallet", "钱包"), ("Backpack", "双肩包")],
    "confectionery": [("Chocolate Box", "巧克力礼盒"), ("Cookies", "曲奇"), ("Candy Tin", "糖果罐"), ("Fudge", "软糖")],
    "health": [
        ("Manuka Honey", "麦卢卡蜂蜜"),
        ("Fish Oil", "鱼油"),
        ("Collagen", "胶原蛋白"),
        ("Vitamin C", "维生素C"),
    ],
    "electronics": [("Earbuds", "耳机"), ("Speaker", "音箱"), ("Camera", "相机"), ("Smartwatch", "智能手表")],
    "eyewear": [("Sunglasses", "太阳镜"), ("Optical Frame", "光学镜架"), ("Aviator", "飞行员墨镜")],
}
ADJ = [
    ("Classic", "经典"),
    ("Radiance", "光采"),
    ("Pure", "纯净"),
    ("Royal", "皇家"),
    ("Silk", "丝绒"),
    ("Midnight", "午夜"),
    ("Golden", "金致"),
    ("Ocean", "海洋"),
    ("Aurora", "极光"),
    ("Heritage", "传承"),
    ("Crystal", "水晶"),
    ("Noble", "尊享"),
    ("Bloom", "花漾"),
    ("Alpine", "高山"),
    ("Velvet", "天鹅绒"),
]
SYL = [
    "Va",
    "Lo",
    "Mi",
    "Ka",
    "Ren",
    "Sol",
    "Bel",
    "Or",
    "Xi",
    "Na",
    "Tor",
    "Lu",
    "Ze",
    "Mar",
    "Quin",
    "Ari",
    "Del",
    "Fen",
    "Sa",
    "Ti",
]
SUFFIX = ["", "e", "ia", "o", " & Co", " Paris", " Milano", " London", " Tokyo"]

MARKETS: dict[str, dict[str, Any]] = {
    "CN-HAINAN": dict(
        fx=7.1,
        share=0.65,
        avail=0.85,
        channels=["island_pickup", "home_delivery"],
        cat_aff={
            "skincare": 1.0,
            "makeup": 0.8,
            "fragrance": 0.6,
            "watches": 0.4,
            "jewelry": 0.4,
            "handbags": 0.5,
            "liquor": 0.1,
            "wine": -0.2,
            "confectionery": -0.3,
            "health": 0.2,
            "electronics": 0.3,
            "eyewear": 0.2,
        },
        countries=[("CN", 0.88), ("HK", 0.04), ("TW", 0.03), ("KR", 0.03), ("RU", 0.02)],
        allowance=100000.0,
    ),
    "NZ-AKL": dict(
        fx=1.7,
        share=0.35,
        avail=0.6,
        channels=["departure_pickup", "arrival_pickup"],
        cat_aff={
            "liquor": 0.9,
            "wine": 0.8,
            "confectionery": 0.7,
            "health": 0.8,
            "fragrance": 0.5,
            "skincare": 0.4,
            "electronics": 0.3,
            "makeup": 0.2,
            "watches": 0.1,
            "jewelry": 0.1,
            "handbags": 0.1,
            "eyewear": 0.2,
        },
        countries=[("NZ", 0.30), ("AU", 0.18), ("CN", 0.25), ("US", 0.10), ("GB", 0.07), ("JP", 0.05), ("KR", 0.05)],
        allowance=0.0,
    ),
}
ZH_COUNTRIES = {"CN", "HK", "TW"}


def _unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-9)


@dataclass
class Truth:
    """Hidden preference model; used by the simulator and by offline evaluation only."""

    user_ids: np.ndarray
    item_ids: np.ndarray
    U: np.ndarray  # user taste vectors
    V: np.ndarray  # item vectors
    pop: np.ndarray  # item intrinsic appeal
    price_z: np.ndarray  # item price position within its category
    user_tier: np.ndarray  # preferred price position
    aff: np.ndarray  # (n_users, n_cat) market-category affinity of the user's market
    item_cat: np.ndarray
    bias: float = -3.4

    def logit(self, u: int, items: np.ndarray) -> np.ndarray:
        return (
            self.U[u] @ self.V[items].T
            + 0.4 * self.pop[items]
            + 0.6 * self.aff[u, self.item_cat[items]]
            - 0.6 * np.abs(self.price_z[items] - self.user_tier[u])
            + self.bias
        )

    def p_click(self, u: int, items: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.logit(u, items)))

    def oracle(self):  # -> Callable[[str, np.ndarray], np.ndarray]
        """(user_id, item_ids) -> ground-truth click probability (NaN for unknown ids)."""
        uix = {str(u): i for i, u in enumerate(self.user_ids)}
        iix = {str(x): i for i, x in enumerate(self.item_ids)}

        def p(user_id: str, item_ids: np.ndarray) -> np.ndarray:
            out = np.full(len(item_ids), np.nan)
            if user_id not in uix:
                return out
            known = np.array([str(x) in iix for x in item_ids], dtype=bool)
            if known.any():
                idx = np.array([iix[str(x)] for x in np.asarray(item_ids)[known]], dtype=int)
                out[known] = self.p_click(uix[user_id], idx)
            return out

        return p

    def save(self, path: Path) -> None:
        np.savez_compressed(path, **{k: getattr(self, k) for k in self.__dataclass_fields__})

    @classmethod
    def load(cls, path: Path) -> Truth:
        z = np.load(path, allow_pickle=True)
        return cls(**{k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files})


@dataclass
class SimResult:
    tables: dict[str, pd.DataFrame]  # canonical: user, item, item_market, category, behavior
    truth: Truth
    meta: dict


def generate(lake_dir: Path, **kw: int) -> dict:
    """Simulate and write canonical tables into a lake (+ hidden truth for evaluation)."""
    sim = simulate(**kw)
    write_lake(sim, lake_dir)
    b = sim.tables["behavior"]
    return dict(
        users=len(sim.tables["user"]),
        items=len(sim.tables["item"]),
        events=len(b),
        **b["event_type"].value_counts().to_dict(),
    )


def write_lake(sim: SimResult, lake_dir: Path) -> None:
    from recsys_core.lake import Lake

    lake = Lake(lake_dir)
    for name, df in sim.tables.items():
        lake.write(name, df)
    lake.write_meta(sim.meta)
    sim.truth.save(Path(lake_dir) / "_truth.npz")


def simulate(
    n_users: int = 6000, n_items: int = 3000, n_brands: int = 120, n_days: int = 30, seed: int = 7
) -> SimResult:
    rng = np.random.default_rng(seed)
    cat_names = [c[0] for c in CATEGORIES]
    n_cat = len(cat_names)
    markets = list(MARKETS)

    # ---- brands
    cat_c = _unit(rng.normal(size=(n_cat, DIM)))
    names: set[str] = set()
    brands = []
    for b in range(n_brands):
        while True:
            nm = "".join(rng.choice(SYL, size=rng.integers(2, 4))).capitalize() + rng.choice(SUFFIX)
            if nm not in names:
                names.add(nm)
                break
        cats = [b % n_cat] + ([int(rng.integers(n_cat))] if rng.random() < 0.3 else [])
        tier = int(rng.choice([1, 2, 3], p=[0.3, 0.45, 0.25]))
        vec = _unit(0.6 * cat_c[cats[0]] + 0.8 * _unit(rng.normal(size=DIM)))
        brands.append(dict(name=nm, cats=cats, tier=tier, vec=vec))

    # ---- items
    cat_w = np.array([3, 3, 2, 2, 1.5, 1, 1, 1, 1.5, 1.5, 1, 1], dtype=float)
    item_cat = rng.choice(n_cat, size=n_items, p=cat_w / cat_w.sum())
    brands_by_cat = {c: [i for i, b in enumerate(brands) if c in b["cats"]] for c in range(n_cat)}
    item_brand = np.array([rng.choice(brands_by_cat[c]) for c in item_cat])
    lo = np.log([CATEGORIES[c][3][0] for c in item_cat])
    hi = np.log([CATEGORIES[c][3][1] for c in item_cat])
    tier_shift = np.array([{1: -0.25, 2: 0.0, 3: 0.3}[brands[b]["tier"]] for b in item_brand])
    u01 = np.clip(rng.beta(2, 2.5, size=n_items) + tier_shift, 0, 1)
    logp = lo + u01 * (hi - lo)
    price_usd = np.round(np.exp(logp), 0)
    price_z = (u01 - 0.45) / 0.2
    V = _unit(
        0.7 * cat_c[item_cat]
        + 0.5 * np.stack([brands[b]["vec"] for b in item_brand])
        + 0.35 * rng.normal(size=(n_items, DIM)) / np.sqrt(DIM)
    )
    pop = rng.normal(0, 1, size=n_items)
    publish_day = np.where(rng.random(n_items) < 0.1, rng.uniform(0, n_days, n_items), rng.uniform(-365, 0, n_items))
    publish_ts = (START_TS + publish_day * DAY).astype(np.int64)
    item_ids = np.array([f"g{i:05d}" for i in range(n_items)])

    titles_en, titles_zh = [], []
    for i in range(n_items):
        cn = cat_names[item_cat[i]]
        a = ADJ[rng.integers(len(ADJ))]
        n = NOUNS[cn][rng.integers(len(NOUNS[cn]))]
        bn = brands[item_brand[i]]["name"]
        titles_en.append(f"{bn} {a[0]} {n[0]}")
        titles_zh.append(f"{bn} {a[1]}{n[1]}")

    items = pd.DataFrame(
        dict(
            item_id=item_ids,
            title_en=titles_en,
            title_zh=titles_zh,
            category=[cat_names[c] for c in item_cat],
            brand=[brands[b]["name"] for b in item_brand],
            price_usd=price_usd,
            publish_ts=publish_ts,
            image=[CATEGORIES[c][2] for c in item_cat],
        )
    )

    # ---- per-market availability
    avail = {}
    im_rows = []
    for m, cfg in MARKETS.items():
        aff = np.array([cfg["cat_aff"][c] for c in cat_names])
        p = np.clip(cfg["avail"] + 0.15 * aff[item_cat], 0.05, 0.98)
        a = rng.random(n_items) < p
        avail[m] = a
        stock = np.where(rng.random(n_items) < 0.03, 0, rng.integers(5, 500, n_items))
        for i in np.flatnonzero(a).tolist():
            im_rows.append(
                (
                    item_ids[i],
                    m,
                    float(np.round(price_usd[i] * cfg["fx"] * rng.uniform(0.95, 1.05), 0)),
                    {"CN-HAINAN": "CNY", "NZ-AKL": "NZD"}[m],
                    int(stock[i]),
                    "|".join(cfg["channels"]),
                )
            )
    item_market = pd.DataFrame(im_rows, columns=["item_id", "market", "price", "currency", "stock", "channels"])
    local_price = {m: np.full(n_items, np.nan) for m in markets}
    for mk, iid, pr in zip(item_market["market"], item_market["item_id"], item_market["price"], strict=True):
        local_price[str(mk)][int(str(iid)[1:])] = float(pr)

    # ---- users
    user_market = rng.choice(markets, size=n_users, p=[MARKETS[m]["share"] for m in markets])
    U = np.zeros((n_users, DIM))
    aff_u = np.zeros((n_users, n_cat))
    countries, langs = [], []
    member_level = rng.choice([0, 1, 2, 3], size=n_users, p=[0.45, 0.3, 0.18, 0.07])
    user_tier = rng.normal(member_level * 0.45 - 0.4, 0.6)
    for u in range(n_users):
        cfg = MARKETS[user_market[u]]
        aff = np.array([cfg["cat_aff"][c] for c in cat_names])
        aff_u[u] = aff
        pc = np.exp(1.5 * aff)
        favs = rng.choice(n_cat, size=rng.integers(1, 4), replace=False, p=pc / pc.sum())
        w = rng.dirichlet(np.ones(len(favs)))
        fav_brand = rng.choice(brands_by_cat[favs[0]])
        vec = (
            (w[:, None] * cat_c[favs]).sum(0)
            + 0.5 * brands[fav_brand]["vec"]
            + 0.3 * rng.normal(size=DIM) / np.sqrt(DIM)
        )
        U[u] = _unit(vec) * rng.lognormal(np.log(2.6), 0.25)
        cs, ps = zip(*cfg["countries"])
        c = rng.choice(cs, p=np.array(ps) / sum(ps))
        countries.append(c)
        langs.append("zh" if c in ZH_COUNTRIES or (c == "SG" and rng.random() < 0.5) else "en")
    user_ids = np.array([f"u{u:05d}" for u in range(n_users)])
    allowance_used = np.where(
        user_market == "CN-HAINAN",
        np.where(
            rng.random(n_users) < 0.1,
            rng.uniform(85000, 99500, n_users),
            np.where(rng.random(n_users) < 0.4, rng.uniform(0, 60000, n_users), 0.0),
        ),
        0.0,
    )
    users = pd.DataFrame(
        dict(
            user_id=user_ids,
            market=user_market,
            country=countries,
            language=langs,
            member_level=member_level,
            register_ts=(START_TS - rng.uniform(0, 900, n_users) * DAY).astype(np.int64),
            allowance_used=np.round(allowance_used, 0),
        )
    )

    truth = Truth(
        user_ids=user_ids,
        item_ids=item_ids,
        U=U,
        V=V,
        pop=pop,
        price_z=price_z,
        user_tier=user_tier,
        aff=aff_u,
        item_cat=item_cat,
    )

    # ---- related-item neighbours used by the incumbent "detail page" logic
    sims = V @ V.T
    np.fill_diagonal(sims, -9)

    # ---- behaviour log (incumbent policy: popular + random + category browsing)
    pop_p = np.exp(0.8 * pop)
    rows = []
    rid = 0
    for u in range(n_users):
        m = user_market[u]
        trip_start = rng.uniform(-3, n_days)
        trip_len = rng.uniform(2, 7)
        n_sess = max(1, rng.poisson(rng.lognormal(np.log(4), 0.6)))
        spent_box = [allowance_used[u]]
        bought_cat = np.zeros(n_cat)
        for _ in range(n_sess):
            day = trip_start + rng.uniform(0, trip_len) if rng.random() < 0.8 else rng.uniform(0, n_days)
            if not 0 <= day < n_days:
                continue
            t = int(START_TS + day * DAY)
            pool = np.flatnonzero(avail[m] & (publish_ts <= t))
            k_pop, k_rand = 8, 6
            pp = pop_p[pool] / pop_p[pool].sum()
            feed = np.concatenate(
                [
                    rng.choice(pool, size=k_pop, replace=False, p=pp),
                    rng.choice(pool, size=k_rand, replace=False),
                    _browse(rng, pool, item_cat, n_cat),
                ]
            ).astype(int)
            feed = np.unique(feed)
            rng.shuffle(feed)
            rid += 1
            req = f"sim-{rid}"
            pos_bias = 1.0 / (1.0 + 0.08 * np.arange(len(feed)))
            p = truth.p_click(u, feed) * pos_bias
            clicks = rng.random(len(feed)) < p
            for pos, i in enumerate(feed):
                rows.append(
                    (req, user_ids[u], item_ids[i], "impression", t + pos, f"demo$##$home$##$feed$##${pos}", m, "", 0.0)
                )
            t += 30
            for i in feed[clicks]:
                t += int(rng.integers(5, 120))
                rows.append((req, user_ids[u], item_ids[i], "click", t, "demo$##$home$##$feed$##$0", m, "", 0.0))
                t = _engage(
                    rng,
                    truth,
                    u,
                    i,
                    m,
                    t,
                    rows,
                    user_ids,
                    item_ids,
                    local_price,
                    MARKETS[m]["allowance"],
                    spent_box,
                    bought_cat,
                    item_cat,
                    req,
                    "home",
                )
                # detail page: related items by the incumbent's similarity logic
                cand = pool[np.argsort(-sims[i, pool])[:30]]
                rel = rng.choice(cand, size=min(6, len(cand)), replace=False)
                rid += 1
                rreq = f"sim-{rid}"
                pr = np.clip(truth.p_click(u, rel) * 1.3, 0, 1)
                rc = rng.random(len(rel)) < pr
                for pos, j in enumerate(rel):
                    rows.append(
                        (
                            rreq,
                            user_ids[u],
                            item_ids[j],
                            "impression",
                            t + pos,
                            f"demo$##$item$##$related$##${pos}",
                            m,
                            item_ids[i],
                            0.0,
                        )
                    )
                for j in rel[rc]:
                    t += int(rng.integers(5, 90))
                    rows.append(
                        (
                            rreq,
                            user_ids[u],
                            item_ids[j],
                            "click",
                            t,
                            "demo$##$item$##$related$##$0",
                            m,
                            item_ids[i],
                            0.0,
                        )
                    )
                    t = _engage(
                        rng,
                        truth,
                        u,
                        j,
                        m,
                        t,
                        rows,
                        user_ids,
                        item_ids,
                        local_price,
                        MARKETS[m]["allowance"],
                        spent_box,
                        bought_cat,
                        item_cat,
                        rreq,
                        "item",
                    )

    behaviors = pd.DataFrame(
        rows,
        columns=[
            "request_id",
            "user_id",
            "item_id",
            "event_type",
            "ts",
            "spm",
            "market",
            "parent_item_id",
            "paid_amount",
        ],
    ).sort_values("ts", kind="stable")

    categories = pd.DataFrame([dict(category=c[0], label_en=c[0], label_zh=c[1], image=c[2]) for c in CATEGORIES])
    meta = dict(source="simulator", start_ts=START_TS, n_days=n_days, seed=seed, n_users=n_users, n_items=n_items)
    return SimResult(
        dict(user=users, item=items, item_market=item_market, category=categories, behavior=behaviors), truth, meta
    )


def _browse(rng, pool, item_cat, n_cat) -> np.ndarray:
    """Incumbent category-browsing slot: 6 random items of a random category."""
    sub = pool[item_cat[pool] == rng.integers(n_cat)]
    return rng.choice(sub, size=6, replace=True) if len(sub) else np.array([], dtype=int)


def _engage(
    rng, truth, u, i, m, t, rows, user_ids, item_ids, local_price, allowance, spent_ref, bought_cat, item_cat, req, page
) -> int:
    """After a click: maybe favorite / add to cart / purchase."""
    lg = truth.logit(u, np.array([i]))[0]
    p_cart = 1 / (1 + np.exp(-(lg + 1.2)))
    spm = f"demo$##${page}$##${'feed' if page == 'home' else 'related'}$##$0"
    if rng.random() < 0.15:
        t += int(rng.integers(3, 30))
        rows.append((req, user_ids[u], item_ids[i], "favorite", t, spm, m, "", 0.0))
    if rng.random() < p_cart:
        t += int(rng.integers(5, 60))
        rows.append((req, user_ids[u], item_ids[i], "add_cart", t, spm, m, "", 0.0))
        price = local_price[m][i]
        ok = allowance <= 0 or spent_ref[0] + price <= allowance
        if ok and rng.random() < 0.45:
            t += int(rng.integers(30, 600))
            rows.append((req, user_ids[u], item_ids[i], "purchase", t, spm, m, "", float(price)))
            spent_ref[0] += price
            bought_cat[item_cat[i]] += 1
    return t

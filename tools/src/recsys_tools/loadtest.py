"""HTTP load test for the predict API (P1 acceptance: 2000 QPS at P99 <= 200 ms).

Closed-loop: `concurrency` virtual clients spread over `procs` processes (so the client is
not the bottleneck), each sending predict requests back to back with a slot mix of
home feed / item related / cart; optionally one behavior write per N predicts.
Optional `rate` turns it into an open-loop test at a fixed request rate."""

from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

MIX = (("home_feed", 0.6), ("item_related", 0.25), ("cart", 0.15))


@dataclass
class Plan:
    url: str
    app_id: str
    spms: dict[str, str]
    users: list[str]
    items: list[str]
    duration: float
    concurrency: int
    warmup: float
    write_every: int
    rate: float | None


def _body(plan: Plan, rng: random.Random) -> dict[str, Any]:
    r, acc, slot = rng.random(), 0.0, "home_feed"
    for name, w in MIX:
        acc += w
        if r < acc:
            slot = name
            break
    if slot not in plan.spms:
        slot = next(iter(plan.spms))
    b: dict[str, Any] = {
        "user": {"uid": rng.choice(plan.users)},
        "context": {"spm": plan.spms[slot] + "$##$0", "feature": {"stringArrayFeature": {}}},
    }
    if slot == "item_related":
        b["parentItem"] = {"id": rng.choice(plan.items)}
    if slot == "cart":
        b["context"]["feature"]["stringArrayFeature"]["cart_items"] = {"values": rng.sample(plan.items, 2)}
    return b


async def _client(plan: Plan, seed: int, out: dict[str, list], deadline: float, t_measure: float, tick: Any) -> None:
    import httpx

    rng = random.Random(seed)
    limits = httpx.Limits(max_connections=1, max_keepalive_connections=1)
    headers = {"Content-Type": "application/json"}
    async with httpx.AsyncClient(base_url=plan.url, limits=limits, timeout=5.0, headers=headers) as c:
        n = 0
        while time.perf_counter() < deadline:
            if tick is not None:
                await tick.get()
            body = _body(plan, rng)
            t0 = time.perf_counter()
            try:
                r = await c.post(f"/predict/api/{plan.app_id}/default", content=json.dumps(body))
                ok = r.status_code == 200 and r.json().get("code") == 200
                degraded = bool(r.json().get("degraded")) if ok else False
            except Exception:
                ok, degraded = False, False
            t1 = time.perf_counter()
            if t0 >= t_measure:
                out["lat"].append(t1 - t0)
                out["err"].append(not ok)
                out["deg"].append(degraded)
            n += 1
            if plan.write_every and n % plan.write_every == 0 and ok:
                items = r.json()["value"]["items"]
                if items:
                    ev = [{"user_id": body["user"]["uid"], "item_id": items[0]["id"], "event_type": "click"}]
                    try:
                        await c.post(f"/data/api/{plan.app_id}/behavior/write", content=json.dumps(ev))
                    except Exception:
                        out["write_err"].append(True)


async def _proc_main(plan: Plan, n_clients: int, seed: int, rate: float | None) -> dict[str, list]:
    out: dict[str, list] = {"lat": [], "err": [], "deg": [], "write_err": []}
    start = time.perf_counter()
    deadline = start + plan.warmup + plan.duration
    t_measure = start + plan.warmup
    tick: asyncio.Queue | None = None
    tasks = []
    if rate:
        tick = asyncio.Queue(maxsize=n_clients * 4)

        async def pacer() -> None:
            assert tick is not None
            step, nxt = 1.0 / rate, time.perf_counter()
            while nxt < deadline:
                nxt += step
                await asyncio.sleep(max(0.0, nxt - time.perf_counter()))
                if not tick.full():
                    tick.put_nowait(1)

        tasks.append(asyncio.create_task(pacer()))
    tasks += [
        asyncio.create_task(_client(plan, seed * 1000 + i, out, deadline, t_measure, tick)) for i in range(n_clients)
    ]
    await asyncio.wait(tasks, timeout=plan.warmup + plan.duration + 10)
    for t in tasks:
        t.cancel()
    return out


def _proc(plan: Plan, n_clients: int, seed: int, rate: float | None, q: Any) -> None:
    q.put(asyncio.run(_proc_main(plan, n_clients, seed, rate)))


def run_loadtest(
    url: str,
    cfg: dict,
    users: list[str],
    items: list[str],
    concurrency: int = 64,
    duration: float = 30,
    procs: int = 4,
    warmup: float = 3,
    write_every: int = 0,
    rate: float | None = None,
) -> dict[str, Any]:
    plan = Plan(
        url,
        cfg["app"]["app_id"],
        {k: v["spm"] for k, v in cfg["slots"].items()},
        users,
        items,
        duration,
        concurrency,
        warmup,
        write_every,
        rate,
    )
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    per = [concurrency // procs + (1 if i < concurrency % procs else 0) for i in range(procs)]
    ps = [
        ctx.Process(target=_proc, args=(plan, n, i, (rate / procs) if rate else None, q))
        for i, n in enumerate(per)
        if n
    ]
    for p in ps:
        p.start()
    parts = [q.get(timeout=warmup + duration + 60) for _ in ps]
    for p in ps:
        p.join(5)
    lat = np.array([x for p in parts for x in p["lat"]]) * 1000
    err = np.array([x for p in parts for x in p["err"]], dtype=bool)
    deg = np.array([x for p in parts for x in p["deg"]], dtype=bool)
    n = len(lat)
    ok_lat = lat[~err] if n else lat
    return {
        "requests": int(n),
        "qps": n / duration if duration else 0.0,
        "errors": int(err.sum()),
        "error_rate": float(err.mean()) if n else 0.0,
        "degraded_rate": float(deg.mean()) if n else 0.0,
        "p50_ms": float(np.percentile(ok_lat, 50)) if len(ok_lat) else float("nan"),
        "p95_ms": float(np.percentile(ok_lat, 95)) if len(ok_lat) else float("nan"),
        "p99_ms": float(np.percentile(ok_lat, 99)) if len(ok_lat) else float("nan"),
        "max_ms": float(ok_lat.max()) if len(ok_lat) else float("nan"),
        "write_errors": sum(len(p["write_err"]) for p in parts),
        "concurrency": concurrency,
        "client_procs": len(ps),
        "duration_s": duration,
        "mode": f"open loop @ {rate:g} req/s" if rate else "closed loop",
    }


def to_markdown(results: list[dict[str, Any]], meta: dict[str, Any]) -> str:
    lines = [
        "# Load test — predict API",
        "",
        ", ".join(f"{k}: **{v}**" for k, v in meta.items()),
        "",
        "Slot mix: " + ", ".join(f"{n} {w:.0%}" for n, w in MIX) + ". Latency is client-side (includes HTTP).",
        "",
        "| mode | concurrency | QPS | p50 ms | p95 ms | p99 ms | max ms | errors | degraded | P1 target |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        ok = "✅" if r["qps"] >= 2000 and r["p99_ms"] <= 200 and r["error_rate"] < 0.001 else "—"
        lines.append(
            f"| {r['mode']} | {r['concurrency']} | {r['qps']:,.0f} | {r['p50_ms']:.1f} | {r['p95_ms']:.1f} "
            f"| {r['p99_ms']:.1f} | {r['max_ms']:.1f} | {r['error_rate']:.2%} | {r['degraded_rate']:.2%} | {ok} |"
        )
    return "\n".join(lines) + "\n"

"""`recsys` command line.

simulate      synthetic data -> lake (and/or a vendor-style raw export with --raw)
backfill      raw export + mapping -> lake, with backfill & quality report
quality       run the data-quality gate on a lake (exit 1 on failure)
baseline      offline baseline report (Hot / Swing / Embedding / GBDT)
train         build all models on the full lake -> artifacts/engine.pkl (+ ranker.onnx)
serve         HTTP API + demo UI (--workers N)
report        T+1 daily report: attribution, funnel by slot/market/variant, A/B readout
ab-sim        online A/B simulation through the production path, then the daily report
consume       Kafka topics -> lake (when [ingest] bus = "kafka")
loadtest      HTTP load test of a running server (P1 target: 2000 QPS, P99 <= 200 ms)
config-check  validate configs/engine.toml
schema-doc    regenerate docs/data-contract.md
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from recsys_core.config import QualityCfg, find_root, load_config


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"written {path}")


def main(argv: list[str] | None = None) -> int:
    root = find_root()
    ap = argparse.ArgumentParser(prog="recsys", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default=None, help="engine config (default: configs/engine.toml)")
    ap.add_argument("--lake", default=str(root / "lake"), help="canonical lake directory")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("simulate")
    s.add_argument("--users", type=int, default=6000)
    s.add_argument("--items", type=int, default=3000)
    s.add_argument("--days", type=int, default=30)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--raw", default=None, help="also/instead write a vendor-style raw export here")
    s.add_argument("--no-lake", action="store_true", help="only write the raw export")

    b = sub.add_parser("backfill")
    b.add_argument("--mapping", required=True)
    b.add_argument("--raw", required=True)
    b.add_argument("--report", default=str(root / "reports" / "backfill.md"))

    q = sub.add_parser("quality")
    q.add_argument("--report", default=str(root / "reports" / "quality.md"))

    bl = sub.add_parser("baseline")
    bl.add_argument("--test-days", type=int, default=5)
    bl.add_argument("--users", type=int, default=1000)
    bl.add_argument("--k", type=int, default=20)
    bl.add_argument("--truth", default=None, help="ground-truth file (synthetic data only) for simulated CTR")
    bl.add_argument("--report", default=str(root / "reports" / "baseline.md"))

    rp = sub.add_parser("report")
    rp.add_argument("--date", required=True, help="business date YYYYMMDD (each market's own time zone)")
    rp.add_argument("--window-h", type=float, default=72, help="last-click attribution window")
    rp.add_argument("--report", default=None, help="default: reports/daily_<date>.md (+ .json)")

    ab = sub.add_parser("ab-sim")
    ab.add_argument("--sessions", type=int, default=6000)
    ab.add_argument("--out", default=str(root / "lake_ab"), help="output lake (recreated)")
    ab.add_argument("--seed", type=int, default=0)
    ab.add_argument("--truth", default=None, help="ground-truth file (default: <lake>/_truth.npz or raw/_truth.npz)")
    ab.add_argument("--report", default=None, help="default: reports/ab_sim_<date>.md (+ .json)")

    cs = sub.add_parser("consume")
    cs.add_argument("--group", default="recsys-lake")
    cs.add_argument("--max-seconds", type=float, default=None, help="stop after this long (default: run forever)")

    lt = sub.add_parser("loadtest")
    lt.add_argument("--url", default="http://127.0.0.1:8000")
    lt.add_argument("--concurrency", default="64", help="comma list, one run each (closed loop)")
    lt.add_argument("--rate", default=None, help="comma list of req/s for open-loop runs (uses max concurrency)")
    lt.add_argument("--duration", type=float, default=20)
    lt.add_argument("--procs", type=int, default=4, help="client processes")
    lt.add_argument("--write-every", type=int, default=0, help="also write one click per N predicts")
    lt.add_argument("--note", default="", help="free text for the report header (e.g. server workers)")
    lt.add_argument("--report", default=str(root / "reports" / "loadtest.md"))

    sub.add_parser("train")
    v = sub.add_parser("serve")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8000)
    v.add_argument("--workers", type=int, default=1, help="processes; use [state] backend = 'redis' when > 1")
    v.add_argument("--log-level", default="info")
    sub.add_parser("config-check")
    sd = sub.add_parser("schema-doc")
    sd.add_argument("--out", default=str(root / "docs" / "data-contract.md"))

    a = ap.parse_args(argv)
    artifact = root / "artifacts" / "engine.pkl"
    lake = Path(a.lake)

    if a.cmd == "config-check":
        cfg = load_config(a.config)
        from rec_server.engine import Engine

        Engine.validate_config(cfg)
        print(
            f"config OK: {len(cfg['markets'])} markets, {len(cfg['slots'])} slots, "
            f"{len(cfg['experiments'])} experiment layers"
        )
        return 0

    if a.cmd == "schema-doc":
        from recsys_core.schema import to_markdown

        _write(Path(a.out), to_markdown())
        return 0

    cfg = load_config(a.config)
    qcfg = QualityCfg(**cfg["quality"])

    if a.cmd == "simulate":
        from .simulator import simulate, write_lake

        sim = simulate(n_users=a.users, n_items=a.items, n_days=a.days, seed=a.seed)
        if not a.no_lake:
            write_lake(sim, lake)
            print(f"lake: {lake} ({len(sim.tables['behavior']):,} events)")
        if a.raw:
            from .raw_export import export_raw

            print(f"raw export: {a.raw}", export_raw(sim, Path(a.raw)))
        return 0

    if a.cmd == "backfill":
        from .backfill import run_backfill

        res = run_backfill(a.mapping, a.raw, lake, qcfg)
        _write(Path(a.report), res.to_markdown())
        print(f"quality gate: {'PASSED' if res.quality.passed else 'FAILED'}")
        return 0 if res.quality.passed else 1

    if a.cmd == "quality":
        from recsys_core.lake import Lake
        from recsys_core.quality import run_checks

        lk = Lake(lake)
        rep = run_checks({t: lk.read(t, numpy=False) for t in ("user", "item", "item_market", "behavior")}, qcfg)
        _write(Path(a.report), rep.to_markdown())
        print(f"quality gate: {'PASSED' if rep.passed else 'FAILED'}")
        return 0 if rep.passed else 1

    if a.cmd == "baseline":
        from recsys_batch.baseline import run

        oracle = None
        truth_path = Path(a.truth) if a.truth else lake / "_truth.npz"
        if truth_path.exists():
            from .simulator import Truth

            oracle = Truth.load(truth_path).oracle()
            print(f"using ground truth {truth_path} for simulated CTR")
        report = run(cfg, str(lake), test_days=a.test_days, n_users=a.users, k=a.k, oracle=oracle)
        _write(Path(a.report), report)
        print(report)
        return 0

    if a.cmd == "report":
        from recsys_batch.report import daily_report

        daily = daily_report(str(lake), cfg, a.date, a.window_h)
        out = Path(a.report or root / "reports" / f"daily_{a.date}.md")
        _write(out, daily.to_markdown())
        _write(out.with_suffix(".json"), daily.to_json())
        print(daily.to_markdown())
        return 0

    if a.cmd == "ab-sim":
        from recsys_batch.report import daily_report

        from .abtest_sim import run_ab_sim
        from .simulator import Truth

        truth = Path(a.truth) if a.truth else lake / "_truth.npz"
        if not truth.exists() and (root / "raw" / "_truth.npz").exists():
            truth = root / "raw" / "_truth.npz"  # lake built by `make p0` from the raw export
        if not truth.exists():
            print(f"{truth} not found: ab-sim needs a simulated lake (recsys simulate / make p0)", file=sys.stderr)
            return 1
        if not artifact.exists():
            print(f"{artifact} not found: run `recsys train` first", file=sys.stderr)
            return 1
        out_lake = Path(a.out)
        date, cfg_ab = run_ab_sim(cfg, artifact, lake, out_lake, Truth.load(truth).oracle(), a.sessions, seed=a.seed)
        daily = daily_report(str(out_lake), cfg_ab, date)
        out = Path(a.report or root / "reports" / f"ab_sim_{date}.md")
        _write(out, daily.to_markdown())
        _write(out.with_suffix(".json"), daily.to_json())
        print(daily.to_markdown())
        return 0

    if a.cmd == "consume":
        from .consume import consume

        return consume(cfg, lake, a.group, a.max_seconds)

    if a.cmd == "loadtest":
        import json
        import platform

        from recsys_core.lake import Lake as _Lake

        from .loadtest import run_loadtest
        from .loadtest import to_markdown as lt_markdown

        lk = _Lake(lake)
        users = lk.read("user")["user_id"].astype(str).tolist() + [f"guest-{i}" for i in range(200)]
        items = lk.read("item")["item_id"].astype(str).tolist()
        runs = []
        concs = [int(x) for x in a.concurrency.split(",")]
        for cc in concs:
            r = run_loadtest(a.url, cfg, users, items, cc, a.duration, a.procs, write_every=a.write_every)
            print(json.dumps(r))
            runs.append(r)
        for rate in [float(x) for x in a.rate.split(",")] if a.rate else []:
            r = run_loadtest(
                a.url, cfg, users, items, max(concs), a.duration, a.procs, write_every=a.write_every, rate=rate
            )
            print(json.dumps(r))
            runs.append(r)
        meta = {"url": a.url, "machine": f"{platform.machine()} · {os.cpu_count()} cores", "note": a.note or "-"}
        _write(Path(a.report), lt_markdown(runs, meta))
        _write(Path(a.report).with_suffix(".json"), json.dumps({"meta": meta, "runs": runs}, indent=2))
        return 0

    if a.cmd == "train":
        from rec_server.data import Tables
        from rec_server.pipeline import build_engine, save_engine

        t = Tables.load(lake)
        eng = build_engine(cfg, t, t.end_ts)
        save_engine(eng, artifact)
        print(f"saved {artifact}")
        if eng.ranker is not None and eng.ranker.onnx_model:
            import json

            from rec_server.features import CATEGORICAL, FEATURES

            onnx_path = artifact.with_name("ranker.onnx")
            onnx_path.write_bytes(eng.ranker.onnx_model)
            spec = {
                "input": "X (float32, [n, len(features)])",
                "output": "p = P(click)",
                "features": FEATURES,
                "categorical": [FEATURES[i] for i in CATEGORICAL],
                "categorical_preprocess": "clip to >= 0",
                "metrics": eng.info.get("ranker", {}),
            }
            _write(artifact.with_name("ranker.json"), json.dumps(spec, indent=2, default=str))
            print(f"saved {onnx_path} ({len(eng.ranker.onnx_model):,} bytes, serving via onnxruntime)")
        return 0

    if a.cmd == "serve":
        import uvicorn

        if a.workers > 1 and cfg["state"]["backend"] == "memory":
            print(
                "warning: --workers > 1 with [state] backend = 'memory': real-time state is per worker", file=sys.stderr
            )
        if a.workers > 1 and not os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
            import shutil

            prom = root / "artifacts" / "prometheus"  # per-worker metric files, aggregated by /metrics
            shutil.rmtree(prom, ignore_errors=True)
            prom.mkdir(parents=True)
            os.environ["PROMETHEUS_MULTIPROC_DIR"] = str(prom)
        os.environ["RECSYS_ARTIFACT"] = str(artifact)
        if a.config:
            os.environ["RECSYS_CONFIG"] = str(a.config)
        uvicorn.run(
            "rec_server.api:app_from_env",
            factory=True,
            host=a.host,
            port=a.port,
            workers=a.workers,
            log_level=a.log_level,
            access_log=False,
        )
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())

"""`recsys` command line.

simulate      synthetic data -> lake (and/or a vendor-style raw export with --raw)
backfill      raw export + mapping -> lake, with backfill & quality report
quality       run the data-quality gate on a lake (exit 1 on failure)
baseline      offline baseline report (Hot / Swing / Embedding / GBDT)
train         build all models on the full lake -> artifacts/engine.pkl
serve         HTTP API + demo UI
config-check  validate configs/engine.toml
schema-doc    regenerate docs/data-contract.md
"""

from __future__ import annotations

import argparse
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

    sub.add_parser("train")
    v = sub.add_parser("serve")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8000)
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

    if a.cmd == "train":
        from rec_server.data import Tables
        from rec_server.pipeline import build_engine, save_engine

        t = Tables.load(lake)
        save_engine(build_engine(cfg, t, t.end_ts), artifact)
        print(f"saved {artifact}")
        return 0

    if a.cmd == "serve":
        import uvicorn

        from rec_server.api import create_app

        uvicorn.run(create_app(artifact, a.config), host=a.host, port=a.port)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())

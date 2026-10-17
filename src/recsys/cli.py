"""Command line: recsys generate | evaluate | train | report | serve."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("recsys")
DEFAULT_ARTIFACTS = Path("artifacts")


def _synthetic(args: argparse.Namespace):
    from .simulator import generate_history, make_world

    world = make_world(args.learners, args.days, args.seed)
    events, _ = generate_history(world, args.days, args.seed + 1)
    return world.catalog, events


def cmd_generate(args: argparse.Namespace) -> None:
    catalog, events = _synthetic(args)
    args.out.mkdir(parents=True, exist_ok=True)
    events.to_csv(args.out / "events.csv", index=False)
    catalog.to_frame().to_csv(args.out / "lessons.csv", index=False)
    log.info("wrote %d events and %d lessons to %s", len(events), len(catalog), args.out)


def cmd_evaluate(args: argparse.Namespace) -> None:
    from .data import catalog_from_events, load_ednet_kt1, load_events_csv, load_implicit_csv
    from .evaluate import run_offline

    if args.data is None:
        catalog, events = _synthetic(args)
        cutoff = args.cutoff if args.cutoff is not None else 0.75 * args.days
    else:
        loader = {"canonical": load_events_csv, "csv": load_implicit_csv, "ednet": load_ednet_kt1}
        events = loader[args.format](args.data)
        catalog = catalog_from_events(events)
        cutoff = args.cutoff if args.cutoff is not None else float(events["t"].quantile(0.8))
    result = run_offline(events, catalog, cutoff)
    with pd.option_context("display.width", 140):
        print(result.table.round(3).to_string())


def cmd_train(args: argparse.Namespace) -> None:
    from .artifacts import train_artifacts
    from .evaluate import OfflineConfig

    catalog, events = _synthetic(args)
    meta = train_artifacts(events, catalog, float(args.days), args.artifacts, OfflineConfig())
    print(json.dumps(meta, indent=2))
    log.info("saved artefacts to %s", args.artifacts)


def cmd_report(args: argparse.Namespace) -> None:
    from .report import write

    study = write(args.out)
    print(study.offline.table.round(3).to_string())
    log.info("wrote %s and charts in %s", args.out / "results.md", args.out / "img")


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from .api import create_app
    from .artifacts import load_artifacts

    if not (args.artifacts / "meta.json").exists():
        raise SystemExit(f"no artefacts in {args.artifacts}: run `recsys train` first")
    uvicorn.run(create_app(load_artifacts(args.artifacts)), host=args.host, port=args.port)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="recsys", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="write a synthetic event log and catalogue as CSV")
    gen.add_argument("--out", type=Path, default=Path("data"))
    ev = sub.add_parser("evaluate", help="offline evaluation (synthetic, or a real log)")
    ev.add_argument("--data", type=Path, help="event CSV or EdNet KT1 folder")
    ev.add_argument("--format", choices=["canonical", "csv", "ednet"], default="canonical")
    ev.add_argument("--cutoff", type=float, help="split time in days")
    train = sub.add_parser("train", help="fit the deployed policy and save artefacts")
    train.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    for p in (gen, ev, train):
        p.add_argument("--learners", type=int, default=2000)
        p.add_argument("--days", type=int, default=56)
        p.add_argument("--seed", type=int, default=0)

    report = sub.add_parser("report", help="write docs/results.md and docs/img/*.png")
    report.add_argument("--out", type=Path, default=Path("docs"))

    serve = sub.add_parser("serve", help="run the API")
    serve.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    gen.set_defaults(func=cmd_generate)
    ev.set_defaults(func=cmd_evaluate)
    train.set_defaults(func=cmd_train)
    report.set_defaults(func=cmd_report)
    serve.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

"""winnow bench summe | winnow run --clips DIR --timeline FILE"""
from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="winnow")
    sub = parser.add_subparsers(dest="command", required=True)

    bench = sub.add_parser("bench", help="run the public-data experiment")
    bench.add_argument("dataset", choices=["summe"])
    bench.add_argument("--data", type=Path, default=Path("data"), help="extracted SumMe folder")
    bench.add_argument("--seed-fraction", type=float, default=0.25)
    bench.add_argument("--only", nargs="*", help="video names, for quick runs")
    bench.add_argument("--deadline-utc", help="HH:MM UTC, stop starting new videos after this and write partial results")

    run = sub.add_parser("run", help="rank one real shoot from a final timeline")
    run.add_argument("--clips", type=Path, required=True)
    run.add_argument("--timeline", type=Path, required=True)
    run.add_argument("--seed-fraction", type=float, default=0.25)
    run.add_argument("--fps", type=float, default=24.0, help="timeline fps, used for EDL timecodes")

    args = parser.parse_args(argv)
    project = Path.cwd()
    if args.command == "bench":
        from .bench import run_bench

        out = run_bench(args.data, project / "cache", project / "runs", args.seed_fraction, args.only, args.deadline_utc)
        print(f"results: {out}")
    else:
        from .pipeline import run_pipeline

        run_pipeline(args.clips, args.timeline, args.seed_fraction, args.fps, project)


if __name__ == "__main__":
    main()

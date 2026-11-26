"""Command line interface for the AneuG-Flow preprocessing: ``aneug <command>``.

Commands:
    scan     write raw_manifest.csv (one row per raw case: files present,
             sizes, oldest modification time) and print totals, missing
             files, the estimated processed size and how close the oldest
             file is to the 30-day scratch purge
    inspect  print, or write with -o, a JSON description of one raw case
    convert  convert cases given as ids, --all or --cases-file; --num-shards
             and --shard-index split the list across Slurm array tasks.
             Completed cases are skipped; a failing case is recorded in
             _failures/ and the remaining cases still run
    verify   compare processed cases value-for-value with the raw data
    features compute geometry_features.npy (wall distance and geodesic
             distances through the fluid from the inlet, outlets and
             aneurysm) for processed
             cases, check the wall files, and keep copies of them; --all
             means every processed case. Failures go to _failures/features/
    index    write index.csv and report complete, failed, incomplete and
             not-yet-processed cases

Paths: --root (default $ANEUG_ROOT, else /scratch.global/$USER/aneug_flow)
holds transient_data/ and processed_data/; --raw-dir and --out-dir override
each one. The exit status is 1 if any case failed. SIGTERM, which Slurm sends
at the time limit, becomes a normal exit so the converter can remove its
temporary directory.

Run as ``aneug ...`` after ``pip install -e .``, or without installing as
``python -m src.data ...`` from the repo root (what scripts/msi/run.sh does).
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
from pathlib import Path
from typing import List, Optional

from . import convert, geometry, layout, manifest, processed, raw

log = logging.getLogger("aneug")


def _paths(args) -> tuple:
    root = Path(args.root) if args.root else layout.default_root()
    raw_dir = Path(args.raw_dir) if args.raw_dir else root / layout.RAW_SUBDIR
    out_dir = Path(args.out_dir) if args.out_dir else root / layout.PROCESSED_SUBDIR
    return raw_dir, out_dir


def _select_cases(args, raw_dir: Path) -> List[str]:
    if args.cases_file:
        ids = manifest.read_case_list(Path(args.cases_file))
    elif args.all:
        ids = [d.name for d in raw.raw_case_dirs(raw_dir)]
    else:
        ids = list(args.cases)
    if not ids:
        raise SystemExit("no cases selected: give case ids, --all, or --cases-file")
    if args.num_shards > 1:
        if not 0 <= args.shard_index < args.num_shards:
            raise SystemExit(f"--shard-index must be in [0, {args.num_shards})")
        ids = ids[args.shard_index :: args.num_shards]
    return ids


def _print_json(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


# --- commands --------------------------------------------------------------


def cmd_scan(args) -> int:
    raw_dir, out_dir = _paths(args)
    cases = manifest.scan_raw(raw_dir)
    path = Path(args.manifest) if args.manifest else out_dir / layout.RAW_MANIFEST_FILE
    manifest.write_manifest(cases, path)
    summary = manifest.summarize(cases)
    summary["raw_dir"] = str(raw_dir)
    summary["manifest"] = str(path)
    _print_json(summary)
    days_left = summary["days_until_oldest_reaches_scratch_purge_age"]
    if days_left is not None and days_left < 7:
        log.warning(
            "oldest raw file is %.1f days old; /scratch.global purges files after %d days",
            summary["oldest_file_age_days"], manifest.SCRATCH_PURGE_DAYS,
        )
    return 0


def cmd_inspect(args) -> int:
    raw_dir, _ = _paths(args)
    from .inspect_raw import inspect_case

    report = inspect_case(raw_dir / args.case)
    text = json.dumps(report, indent=2, default=str)
    if args.output:
        Path(args.output).write_text(text + "\n")
        log.info("wrote %s", args.output)
    else:
        print(text)
    return 0


def cmd_convert(args) -> int:
    raw_dir, out_dir = _paths(args)
    ids = _select_cases(args, raw_dir)
    log.info("converting %d case(s) from %s into %s", len(ids), raw_dir, out_dir)
    counts = {"converted": 0, "skipped": 0, "failed": 0}
    for i, cid in enumerate(ids, 1):
        try:
            res = convert.convert_case(
                raw_dir / cid,
                out_dir,
                overwrite=args.overwrite,
                coord_atol=args.coord_atol,
                allow_nonfinite=args.allow_nonfinite,
                time_chunk=args.time_chunk,
            )
        except Exception as exc:
            counts["failed"] += 1
            path = convert.record_failure(out_dir, cid, exc)
            log.error("[%d/%d] %s FAILED: %s: %s (details: %s)", i, len(ids), cid, type(exc).__name__, exc, path)
            if args.fail_fast:
                break
            continue
        counts[res.status] += 1
        log.info("[%d/%d] %s %s%s", i, len(ids), cid, res.status,
                 f" in {res.seconds:.1f}s" if res.status == "converted" else "")
    log.info("done: %s", counts)
    return 1 if counts["failed"] else 0


def cmd_verify(args) -> int:
    raw_dir, out_dir = _paths(args)
    if args.all and not args.cases_file:
        args.cases = processed.list_cases(out_dir)
        args.all = False
    ids = _select_cases(args, raw_dir)
    failed = 0
    for cid in ids:
        try:
            result = convert.verify_case(raw_dir / cid, out_dir / cid, time_chunk=args.time_chunk)
            log.info("%s OK %s", cid, result)
        except Exception as exc:
            failed += 1
            log.error("%s FAILED: %s: %s", cid, type(exc).__name__, exc)
    log.info("verified %d case(s), %d failed", len(ids), failed)
    return 1 if failed else 0


def cmd_features(args) -> int:
    raw_dir, out_dir = _paths(args)
    if args.all and not args.cases_file:
        args.cases = processed.list_cases(out_dir)
        args.all = False
    ids = _select_cases(args, raw_dir)
    log.info("geometry features for %d case(s) in %s", len(ids), out_dir)
    counts = {"computed": 0, "skipped": 0, "failed": 0}
    for i, cid in enumerate(ids, 1):
        try:
            res = geometry.features_case(out_dir / cid, raw_dir / cid, overwrite=args.overwrite)
        except Exception as exc:
            counts["failed"] += 1
            path = geometry.record_failure(out_dir, cid, exc)
            log.error("[%d/%d] %s FAILED: %s: %s (details: %s)", i, len(ids), cid, type(exc).__name__, exc, path)
            if args.fail_fast:
                break
            continue
        geometry.clear_failure(out_dir, cid)
        counts[res.status] += 1
        extra = f" in {res.seconds:.1f}s, {res.num_warnings} warning(s)" if res.status == "computed" else ""
        log.info("[%d/%d] %s %s%s", i, len(ids), cid, res.status, extra)
    log.info("done: %s", counts)
    return 1 if counts["failed"] else 0


def cmd_index(args) -> int:
    raw_dir, out_dir = _paths(args)
    raw_ids = [d.name for d in raw.raw_case_dirs(raw_dir)] if raw_dir.is_dir() else None
    summary = processed.build_index(out_dir, raw_ids)
    if "not_yet_processed" in summary and len(summary["not_yet_processed"]) > 20:
        n = len(summary["not_yet_processed"])
        summary["not_yet_processed"] = summary["not_yet_processed"][:20] + [f"... {n - 20} more"]
    if len(summary["without_geometry"]) > 20:
        n = len(summary["without_geometry"])
        summary["without_geometry"] = summary["without_geometry"][:20] + [f"... {n - 20} more"]
    _print_json(summary)
    return 1 if summary["failed"] or summary["geometry_failed"] else 0


# --- parser ----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", help=f"dataset root (default: ${layout.ROOT_ENV_VAR} or /scratch.global/$USER/aneug_flow)")
    common.add_argument("--raw-dir", help=f"raw cases (default: <root>/{layout.RAW_SUBDIR})")
    common.add_argument("--out-dir", help=f"processed cases (default: <root>/{layout.PROCESSED_SUBDIR})")
    common.add_argument("-v", "--verbose", action="store_true")

    select = argparse.ArgumentParser(add_help=False)
    select.add_argument("cases", nargs="*", help="case ids, e.g. stable_0")
    select.add_argument("--all", action="store_true", help="every case directory in the raw dir")
    select.add_argument("--cases-file", help="manifest CSV (case_id column) or text file of case ids")
    select.add_argument("--num-shards", type=int, default=1, help="split the selection into N shards")
    select.add_argument("--shard-index", type=int, default=0, help="which shard to process (0-based)")
    select.add_argument("--time-chunk", type=int, default=8, help="snapshots read per step")

    p = argparse.ArgumentParser(prog="aneug", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("scan", parents=[common], help="inventory the raw cases and write a manifest")
    s.add_argument("--manifest", help=f"output CSV (default: <out-dir>/{layout.RAW_MANIFEST_FILE})")
    s.set_defaults(func=cmd_scan)

    s = sub.add_parser("inspect", parents=[common], help="describe the files of one raw case")
    s.add_argument("case")
    s.add_argument("-o", "--output", help="write JSON here instead of stdout")
    s.set_defaults(func=cmd_inspect)

    s = sub.add_parser("convert", parents=[common, select], help="raw cases -> processed layout")
    s.add_argument("--overwrite", action="store_true", help="replace already processed cases")
    s.add_argument("--coord-atol", type=float, default=0.0,
                   help="allowed coordinate change across snapshots (default: exact)")
    s.add_argument("--allow-nonfinite", action="store_true",
                   help="keep cases with NaN/Inf solution values (recorded in metadata)")
    s.add_argument("--fail-fast", action="store_true", help="stop at the first failing case")
    s.set_defaults(func=cmd_convert)

    s = sub.add_parser("verify", parents=[common, select], help="re-check processed cases against raw")
    s.set_defaults(func=cmd_verify)

    s = sub.add_parser("features", parents=[common, select],
                       help="geometry features (wall distance, volume geodesics) for processed cases")
    s.add_argument("--overwrite", action="store_true", help="recompute cases that already have features")
    s.add_argument("--fail-fast", action="store_true", help="stop at the first failing case")
    s.set_defaults(func=cmd_features)

    s = sub.add_parser("index", parents=[common], help="write index.csv and report progress")
    s.set_defaults(func=cmd_index)
    return p


def _exit_on_sigterm(signum, frame):
    # Slurm sends SIGTERM at the time limit; raising here lets the converter
    # remove its temporary directory before exiting.
    raise SystemExit(128 + signum)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    signal.signal(signal.SIGTERM, _exit_on_sigterm)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

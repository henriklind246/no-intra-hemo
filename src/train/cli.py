"""Command line for training: ``python -m src.train <command>``.

Commands:
    train     start a run: ``train --config configs/mlp_baseline.json [--flag value ...]``;
              every TrainConfig field is a flag (``--batch-size 8``, ``--no-amp``)
    resume    continue an interrupted run: ``resume runs/mlp_baseline [--steps N]``
    evaluate  full-field metrics of a run's checkpoint on a split part:
              ``evaluate runs/mlp_baseline --part test [--checkpoint best]``

Exit status: 0 when done, 75 when the run was interrupted (SIGTERM, Ctrl-C)
and can be resumed, 1 on errors.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from .config import TrainConfig

EXIT_INTERRUPTED = 75
log = logging.getLogger("aneug.train")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.train", description=__doc__.split("\n")[0])
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("train", help="start a new run")
    s.add_argument("--config", help="JSON file with TrainConfig fields (flags override it)")
    TrainConfig.add_arguments(s)

    s = sub.add_parser("resume", help="continue an interrupted run from its last.pt")
    s.add_argument("run_dir")
    s.add_argument("--steps", type=int, help="new total number of steps")

    s = sub.add_parser("evaluate", help="full-field metrics of a checkpoint")
    s.add_argument("run_dir")
    s.add_argument("--checkpoint", default="best", help="best or last (default best)")
    s.add_argument("--part", default="test", choices=["train", "val", "test"])
    s.add_argument("--snapshot-stride", type=int, default=1, help="every n-th snapshot (default: all)")
    s.add_argument("--max-cases", type=int, help="only the first n cases of the part")
    s.add_argument("--data-root", help="processed_data (default: the one the run trained on)")
    s.add_argument("--device", default="auto")
    s.add_argument("--num-workers", type=int, default=4)
    s.add_argument("--chunk", type=int, help="points per chunk (default: the run's eval_chunk)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    from . import trainer  # imports torch

    try:
        if args.command == "train":
            base = json.loads(Path(args.config).read_text()) if args.config else {}
            result = trainer.train(TrainConfig.from_namespace(args, base))
        elif args.command == "resume":
            result = trainer.resume(Path(args.run_dir), steps=args.steps)
        else:
            trainer.evaluate_run(Path(args.run_dir), checkpoint=args.checkpoint, part=args.part,
                                 snapshot_stride=args.snapshot_stride, max_cases=args.max_cases,
                                 data_root=args.data_root, device=args.device, num_workers=args.num_workers,
                                 chunk=args.chunk)
            return 0
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        return 1
    return EXIT_INTERRUPTED if result["status"] == "interrupted" else 0


if __name__ == "__main__":
    sys.exit(main())

"""Training configuration: one flat dataclass, settable from JSON and the command line.

``TrainConfig.from_args`` reads ``--config file.json`` (if given) for
defaults, then every field as a ``--flag`` (``--batch-size 8``,
``--amp/--no-amp``). The resolved config is saved as ``config.json`` in the
run directory, and a run is resumed from that file.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import typing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class TrainConfig:
    # --- run and data ---
    name: str = "mlp_baseline"
    out_dir: str = "runs"
    data_root: Optional[str] = None  # processed_data; default: $ANEUG_ROOT/processed_data
    split: str = "splits/default.json"
    normalization: Optional[str] = None  # default: next to the split
    # --- model (mlp or linearno; width/depth/activation/time_harmonics are shared) ---
    model: str = "mlp"
    width: int = 256
    depth: int = 6
    activation: str = "gelu"
    time_harmonics: int = 4
    fourier_features: int = 0  # mlp only
    fourier_scale: float = 1.0  # mlp only
    heads: int = 8  # linearno only, from here
    mlp_ratio: int = 2
    key_ratio: int = 1
    dropout: float = 0.0
    geometry_injection: bool = True  # early FiLM geometry memory injection
    injection_blocks: Optional[int] = None  # inject after the first n blocks; default: half of depth
    checkpoint_blocks: bool = True
    # --- training data ---
    points: int = 32768  # points per (case, snapshot) item
    batch_size: int = 8  # items per step
    snapshots_per_item: int = 1
    num_workers: int = 8
    max_open_cases: int = 32  # per worker
    # --- optimization: AdamW, then warmup_cosine or onecycle ---
    steps: int = 50000
    lr: float = 1e-3  # peak learning rate (OneCycleLR's max_lr)
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    scheduler: str = "warmup_cosine"
    min_lr: float = 1e-5  # warmup_cosine: final learning rate
    warmup_steps: int = 1000  # warmup_cosine
    warmup_fraction: float = 0.05  # onecycle: pct_start
    onecycle_div_factor: float = 25.0  # onecycle: initial lr = lr / div_factor
    onecycle_final_div_factor: float = 1e4  # onecycle: final lr = initial lr / final_div_factor
    loss: str = "mse"  # mse or huber, on the normalized outputs
    huber_delta: float = 1.0
    pressure_weight: float = 1.0
    # --- evaluation and logging ---
    eval_every: int = 2500
    eval_snapshot_stride: int = 10
    eval_snapshot_offset: int = 2  # with stride 10: snapshots 2, 12 (peak inflow), 22, ...
    eval_max_cases: Optional[int] = None
    eval_chunk: int = 262144
    log_every: int = 50
    checkpoint_every: int = 1000
    # --- system ---
    seed: int = 0
    device: str = "auto"  # auto, cpu, cuda, cuda:1, mps
    amp: bool = True  # bfloat16 autocast on CUDA

    def __post_init__(self):
        if self.loss not in ("mse", "huber"):
            raise ValueError(f"loss must be mse or huber, not {self.loss!r}")
        if self.scheduler not in ("warmup_cosine", "onecycle"):
            raise ValueError(f"scheduler must be warmup_cosine or onecycle, not {self.scheduler!r}")
        if self.model not in ("mlp", "linearno"):
            raise ValueError(f"model must be mlp or linearno, not {self.model!r}")
        if not 0 < self.warmup_fraction < 1:
            raise ValueError("warmup_fraction must be in (0, 1)")
        for k in ("points", "batch_size", "snapshots_per_item", "steps", "eval_every", "log_every",
                  "checkpoint_every", "eval_snapshot_stride", "eval_chunk"):
            if getattr(self, k) < 1:
                raise ValueError(f"{k} must be at least 1")

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.name

    def model_config(self) -> dict:
        if self.model == "linearno":
            return dict(n_hidden=self.width, n_layers=self.depth, n_head=self.heads, mlp_ratio=self.mlp_ratio,
                        key_ratio=self.key_ratio, act=self.activation, dropout=self.dropout,
                        time_harmonics=self.time_harmonics, geometry_injection=self.geometry_injection,
                        injection_blocks=self.injection_blocks, checkpoint_blocks=self.checkpoint_blocks)
        return dict(width=self.width, depth=self.depth, activation=self.activation,
                    time_harmonics=self.time_harmonics, fourier_features=self.fourier_features,
                    fourier_scale=self.fourier_scale, fourier_seed=self.seed)

    def eval_snapshots(self, num_snapshots: int) -> List[int]:
        return list(range(self.eval_snapshot_offset % self.eval_snapshot_stride, num_snapshots,
                          self.eval_snapshot_stride))

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> "TrainConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(d) - names)
        if unknown:
            raise ValueError(f"unknown config keys: {unknown}")
        return cls(**d)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        hints = typing.get_type_hints(cls)
        for f in dataclasses.fields(cls):
            flag = "--" + f.name.replace("_", "-")
            tp = hints[f.name]
            if typing.get_origin(tp) is typing.Union:  # Optional[X]
                tp = next(a for a in typing.get_args(tp) if a is not type(None))
            if tp is bool:
                parser.add_argument(flag, dest=f.name, action=argparse.BooleanOptionalAction, default=None)
            else:
                parser.add_argument(flag, dest=f.name, type=tp, default=None, metavar=tp.__name__.upper())

    @classmethod
    def from_namespace(cls, args: argparse.Namespace, base: Optional[dict] = None) -> "TrainConfig":
        d = dict(base or {})
        for f in dataclasses.fields(cls):
            v = getattr(args, f.name, None)
            if v is not None:
                d[f.name] = v
        return cls.from_dict(d)

"""Training loop, checkpoints and evaluation for the flow-field operators.

``train(cfg)`` creates ``<out_dir>/<name>/`` and trains for ``cfg.steps``
optimizer steps; ``resume(run_dir)`` continues an interrupted run from its
``last.pt``. A run directory is self-contained:

    config.json          the resolved TrainConfig
    split.json           copy of the split it was trained on
    normalization.json   copy of the normalization it used
    metrics.jsonl        one line per log interval and per evaluation
    train.log            the console log
    last.pt, best.pt     checkpoints (best = lowest validation score)
    status.json          finished / interrupted, step, best score
    eval_<part>_<ckpt>.json / .csv   from ``evaluate_run``

Training draws batches of ``batch_size`` (case, snapshot) items with
``points`` random points each (``SampledPoints`` + ``EpochSampler``), so a
step sees ``batch_size * points`` points from as many different fields. The
loss is the MSE (or Huber loss) of the normalized outputs, with
``pressure_weight`` on the pressure channel. Optimizer: AdamW; learning rate
per step from ``make_optimizer`` (linear warmup + cosine decay, or OneCycleLR).

Validation runs every ``eval_every`` steps on full fields of the validation
cases at every ``eval_snapshot_stride``-th snapshot (``FullField``), and the
checkpoint with the lowest ``score`` (see metrics.py) is kept as best.pt.
Each field is predicted as a whole (``predict_field``): for LinearNO every
point attends to all points of its field, accumulated chunk by chunk.

On SIGTERM (Slurm sends it before the time limit, given ``--signal``) or
Ctrl-C the loop stops after the current step, saves last.pt and returns
status "interrupted"; ``resume`` then continues at exactly the next batch.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import shutil
import signal
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data import layout
from ..data.dataset import EpochSampler, FullField, SampledPoints
from ..data.normalize import Normalization, normalization_path
from ..data.processed import ProcessedCase
from ..data.provenance import provenance
from ..data.splits import Split
from ..models import build_model, predict_field
from .config import TrainConfig
from .metrics import FieldMetrics

log = logging.getLogger("aneug.train")


# --- setup -------------------------------------------------------------------


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def make_optimizer(model: torch.nn.Module, cfg: TrainConfig):
    """AdamW on every parameter, and the per-step learning-rate schedule.

    warmup_cosine: linear warmup over ``warmup_steps`` to ``lr``, then cosine
    decay to ``min_lr`` at ``steps``.
    onecycle: OneCycleLR with ``max_lr = lr`` over ``steps`` steps, warmup for
    ``warmup_fraction`` of them from ``lr / onecycle_div_factor``, cosine
    annealing down to that divided by ``onecycle_final_div_factor``, and no
    momentum cycling.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                            weight_decay=cfg.weight_decay)
    if cfg.scheduler == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=cfg.lr, total_steps=cfg.steps, pct_start=cfg.warmup_fraction, anneal_strategy="cos",
            cycle_momentum=False, div_factor=cfg.onecycle_div_factor,
            final_div_factor=cfg.onecycle_final_div_factor)
    else:
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: _lr_factor(s, cfg))
    return opt, sched


def _lr_factor(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return (step + 1) / cfg.warmup_steps
    span = max(cfg.steps - cfg.warmup_steps, 1)
    progress = min((step - cfg.warmup_steps) / span, 1.0)
    floor = cfg.min_lr / cfg.lr
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def _ignore_sigterm(_worker_id: int) -> None:
    # Only the main process handles SIGTERM, so workers survive until it has saved.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


def _loader_kwargs(num_workers: int, device: torch.device) -> dict:
    kw = dict(num_workers=num_workers, pin_memory=device.type == "cuda")
    if num_workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=4, worker_init_fn=_ignore_sigterm)
    return kw


def _to(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def make_model(name: str, model_config: dict, norm: Normalization) -> torch.nn.Module:
    return build_model(name, num_point_inputs=len(norm.input_channels), num_flow_split=norm.num_flow_split,
                       **model_config)


def loss_fn(cfg: TrainConfig, device: torch.device):
    w = torch.tensor([1.0, 1.0, 1.0, cfg.pressure_weight], device=device)
    w = w / w.mean()

    def f(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if cfg.loss == "huber":
            e = torch.nn.functional.huber_loss(pred, target, reduction="none", delta=cfg.huber_delta)
        else:
            e = (pred - target) ** 2
        return (e * w).mean()

    return f


# --- evaluation ----------------------------------------------------------------


@torch.no_grad()
def evaluate(model: torch.nn.Module, dataset: FullField, device: torch.device, *, amp: bool = True,
             num_workers: int = 4) -> FieldMetrics:
    """Predict every chunk of ``dataset`` and accumulate the metrics per field."""
    metrics = FieldMetrics(dataset.case_ids, dataset.normalization.target_scale)
    loader = DataLoader(dataset, batch_size=None, **{k: v for k, v in _loader_kwargs(num_workers, device).items()
                                                      if k != "persistent_workers"})
    was_training = model.training
    model.eval()
    field, key = [], None

    def flush():
        # One whole field: models with attention see all of its chunks at once.
        with _autocast(device, amp):
            preds = predict_field(model, [c["inputs"][None] for c in field], field[0]["flow_split"][None],
                                  field[0]["time"][None])
        for c, pred in zip(field, preds):
            target = c["targets"][None]
            if device.type == "mps":  # no float64 on MPS
                pred, target = pred.cpu(), target.cpu()
            metrics.update(pred.float(), target, c["case"][None], c["snapshots"][None])

    # FullField yields the chunks of each (case, snapshot) consecutively.
    for item in loader:
        k = (int(item["case"]), tuple(item["snapshots"].tolist()))
        if field and k != key:
            flush()
            field = []
        field.append(_to(item, device))
        key = k
    if field:
        flush()
    model.train(was_training)
    return metrics


# --- checkpoints -----------------------------------------------------------------


def _save(path: Path, obj) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _write_json(path: Path, obj) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    return torch.load(path, map_location=device, weights_only=True)


class _StopFlag:
    def __init__(self):
        self.requested = False
        self._previous = None

    def __enter__(self):
        def handler(signum, frame):
            log.warning("signal %d: stopping after this step", signum)
            self.requested = True
        try:
            self._previous = signal.signal(signal.SIGTERM, handler)
        except ValueError:  # not the main thread
            self._previous = None
        return self

    def __exit__(self, *exc):
        if self._previous is not None:
            signal.signal(signal.SIGTERM, self._previous)
        return False


def _duration(seconds: float) -> str:
    m = int(round(seconds / 60))
    return f"{m // 60}h{m % 60:02d}m"


def _add_file_log(run_dir: Path) -> logging.Handler:
    h = logging.FileHandler(run_dir / "train.log")
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S"))
    logging.getLogger("aneug").addHandler(h)
    return h


# --- training -----------------------------------------------------------------


def train(cfg: TrainConfig) -> dict:
    """Start a new run in ``cfg.run_dir``."""
    run_dir = cfg.run_dir
    if (run_dir / "config.json").exists():
        raise FileExistsError(f"{run_dir} already holds a run; resume it, or choose another --name")
    data_root = Path(cfg.data_root) if cfg.data_root else layout.default_root() / layout.PROCESSED_SUBDIR
    split_file = Path(cfg.split)
    norm_file = Path(cfg.normalization) if cfg.normalization else normalization_path(split_file)
    Split.load(split_file)  # validate before creating anything
    Normalization.load(norm_file)
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(split_file, run_dir / "split.json")
    shutil.copyfile(norm_file, run_dir / "normalization.json")
    cfg.data_root = str(data_root.resolve())
    cfg.save(run_dir / "config.json")
    return _run(run_dir, cfg, resume=False)


def resume(run_dir: Path, *, steps: Optional[int] = None) -> dict:
    """Continue the run in ``run_dir`` from last.pt (optionally with a new total ``steps``)."""
    run_dir = Path(run_dir)
    cfg = TrainConfig.load(run_dir / "config.json")
    if steps is not None and steps != cfg.steps:
        if cfg.scheduler == "onecycle":
            raise ValueError("OneCycleLR has a fixed length: start a new run to train for a different number "
                             "of steps")
        cfg.steps = steps
        cfg.save(run_dir / "config.json")
    if not (run_dir / "last.pt").exists():
        log.warning("%s has no last.pt; starting it from step 0", run_dir)
        return _run(run_dir, cfg, resume=False)
    return _run(run_dir, cfg, resume=True)


def _run(run_dir: Path, cfg: TrainConfig, *, resume: bool) -> dict:
    handler = _add_file_log(run_dir)
    try:
        return _loop(run_dir, cfg, resume=resume)
    finally:
        logging.getLogger("aneug").removeHandler(handler)
        handler.close()


def _loop(run_dir: Path, cfg: TrainConfig, *, resume: bool) -> dict:
    device = resolve_device(cfg.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    split = Split.load(run_dir / "split.json")
    norm = Normalization.load(run_dir / "normalization.json")
    root = Path(cfg.data_root)

    model = make_model(cfg.model, cfg.model_config(), norm).to(device)
    opt, sched = make_optimizer(model, cfg)
    step, best = 0, {"score": float("inf"), "step": None}
    if resume:
        ckpt = load_checkpoint(run_dir / "last.pt", device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        step, best = int(ckpt["step"]), dict(ckpt["best"])
        log.info("resumed %s at step %d (best score %.4g at step %s)", run_dir, step, best["score"], best["step"])
    n_params = sum(p.numel() for p in model.parameters())
    log.info("model %s with %d parameters on %s", cfg.model, n_params, device)

    train_ds = SampledPoints(split, "train", root, norm, points=cfg.points, snapshots=cfg.snapshots_per_item,
                             seed=cfg.seed, max_open_cases=cfg.max_open_cases)
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, drop_last=True,
                        sampler=EpochSampler(len(train_ds), seed=cfg.seed, start=step * cfg.batch_size),
                        **_loader_kwargs(cfg.num_workers, device))
    val_ds = None
    if split.ids("val"):
        val_ids = split.ids("val")[: cfg.eval_max_cases] if cfg.eval_max_cases else None
        val_ds = FullField(split, "val", root, norm, chunk=cfg.eval_chunk, cases=val_ids,
                           snapshots=cfg.eval_snapshots(train_ds.num_snapshots))
        log.info("validation: %d cases x %d snapshots, full fields", len(val_ds.case_ids),
                 len(cfg.eval_snapshots(train_ds.num_snapshots)))
    else:
        log.warning("the split has no validation cases: no evaluation, best.pt is not written")
    log.info("training: %d cases x %d snapshots, %d items of %d points per step, %d steps",
             len(train_ds.case_ids), train_ds.num_snapshots, cfg.batch_size, cfg.points, cfg.steps)

    loss_of = loss_fn(cfg, device)
    metrics_log = open(run_dir / "metrics.jsonl", "a")

    def record(obj):
        metrics_log.write(json.dumps(obj) + "\n")
        metrics_log.flush()

    def checkpoint(name: str):
        _save(run_dir / name, {
            "model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
            "step": step, "best": best, "model_name": cfg.model, "model_config": model.config,
            "config": cfg.to_dict(),
        })

    eval_seconds = [0.0]  # duration of the last evaluation, for the ETA

    def evals_left() -> int:  # evaluations after the current step, up to and including the last step
        if val_ds is None:
            return 0
        e = cfg.eval_every
        return cfg.steps // e - step // e + (1 if cfg.steps % e else 0)

    def run_eval():
        t = time.time()
        summary = evaluate(model, val_ds, device, amp=cfg.amp, num_workers=min(cfg.num_workers, 4)).summary()
        summary.pop("per_snapshot", None)
        eval_seconds[0] = time.time() - t
        record({"step": step, "val": summary, "eval_seconds": eval_seconds[0]})
        log.info("step %d val: score %.4f, velocity rel L2 %.4f, pressure rel L2 %.4f (centered %.4f), "
                 "%.0f s", step, summary["score"], summary["velocity_rel_l2"], summary["pressure_rel_l2"],
                 summary["pressure_centered_rel_l2"], time.time() - t)
        if summary["score"] < best["score"]:
            best.update(score=float(summary["score"]), step=step)
            checkpoint("best.pt")

    status = "finished"
    it = iter(loader)
    loss_sum = torch.zeros((), device=device)
    n_loss, t_data, t_last = 0, 0.0, time.perf_counter()
    points_per_step = cfg.batch_size * cfg.points * cfg.snapshots_per_item
    model.train()
    try:
        with _StopFlag() as stop:
            while step < cfg.steps:
                if stop.requested:
                    status = "interrupted"
                    break
                t0 = time.perf_counter()
                b = _to(next(it), device)
                t_data += time.perf_counter() - t0
                with _autocast(device, cfg.amp):
                    pred = model(b["inputs"], b["flow_split"], b["time"])
                loss = loss_of(pred.float(), b["targets"])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
                sched.step()
                step += 1
                loss_sum += loss.detach()
                n_loss += 1

                if step % cfg.log_every == 0 or step == cfg.steps:
                    mean_loss = float(loss_sum) / n_loss
                    if not math.isfinite(mean_loss):
                        raise FloatingPointError(f"loss is {mean_loss} at step {step}")
                    elapsed = time.perf_counter() - t_last
                    # remaining training steps at the recent rate, plus the evaluations still due
                    eta = (cfg.steps - step) * elapsed / n_loss + evals_left() * eval_seconds[0]
                    rec = {"step": step, "loss": mean_loss, "lr": sched.get_last_lr()[0],
                           "points_per_s": n_loss * points_per_step / elapsed, "data_wait": t_data / elapsed,
                           "epoch": step * cfg.batch_size / len(train_ds), "eta_s": eta}
                    mem = ""
                    if device.type == "cuda":
                        rec["gpu_mem_gb"] = torch.cuda.max_memory_allocated(device) / 2**30
                        mem = f"  peak mem {rec['gpu_mem_gb']:.1f} GB"
                    record(rec)
                    log.info("step %d/%d loss %.4g lr %.2e  %.2fM points/s  data wait %.0f%%  epoch %.2f%s  eta %s",
                             step, cfg.steps, mean_loss, rec["lr"], rec["points_per_s"] / 1e6,
                             100 * rec["data_wait"], rec["epoch"], mem, _duration(eta))
                    loss_sum.zero_()
                    n_loss, t_data, t_last = 0, 0.0, time.perf_counter()
                if val_ds is not None and (step % cfg.eval_every == 0 or step == cfg.steps):
                    run_eval()
                    t_last = time.perf_counter()
                if step % cfg.checkpoint_every == 0:
                    checkpoint("last.pt")
    except KeyboardInterrupt:
        log.warning("interrupted: saving last.pt")
        status = "interrupted"
    except BaseException:
        metrics_log.close()  # no checkpoint: the state may be broken (e.g. a non-finite loss)
        raise
    checkpoint("last.pt")
    metrics_log.close()
    result = {"status": status, "step": step, "steps": cfg.steps, "best_score": best["score"],
              "best_step": best["step"], "run_dir": str(run_dir), "provenance": provenance()}
    _write_json(run_dir / "status.json", result)
    log.info("%s at step %d; best validation score %.4g at step %s", status, step, best["score"], best["step"])
    return result


# --- evaluation of a finished run ---------------------------------------------


def evaluate_run(run_dir: Path, *, checkpoint: str = "best", part: str = "test", snapshot_stride: int = 1,
                 max_cases: Optional[int] = None, data_root: Optional[str] = None, device: str = "auto",
                 num_workers: int = 4, chunk: Optional[int] = None) -> dict:
    """Evaluate a run's checkpoint on full fields of one split part; writes eval_<part>_<checkpoint>.json/.csv."""
    run_dir = Path(run_dir)
    cfg = TrainConfig.load(run_dir / "config.json")
    split = Split.load(run_dir / "split.json")
    norm = Normalization.load(run_dir / "normalization.json")
    dev = resolve_device(device)
    ckpt = load_checkpoint(run_dir / f"{checkpoint}.pt", dev)
    model = build_model(ckpt["model_name"], **ckpt["model_config"]).to(dev)
    model.load_state_dict(ckpt["model"])
    root = Path(data_root or cfg.data_root)
    ids = split.ids(part)[:max_cases] if max_cases else None
    T = ProcessedCase(root / (ids or split.ids(part))[0]).num_snapshots
    ds = FullField(split, part, root, norm, chunk=chunk or cfg.eval_chunk, cases=ids,
                   snapshots=list(range(0, T, snapshot_stride)))
    t = time.time()
    log.info("evaluating %s (step %d) on %d %s cases x %d snapshots", checkpoint, ckpt["step"],
             len(ds.case_ids), part, len(range(0, T, snapshot_stride)))
    metrics = evaluate(model, ds, dev, amp=cfg.amp, num_workers=num_workers)
    summary = metrics.summary(worst=10)
    out = {"checkpoint": checkpoint, "step": int(ckpt["step"]), "part": part, "snapshot_stride": snapshot_stride,
           "seconds": time.time() - t, **summary, "provenance": provenance()}
    stem = run_dir / f"eval_{part}_{checkpoint}"
    _write_json(stem.with_suffix(".json"), out)
    metrics.write_csv(stem.with_suffix(".csv"))
    log.info("%s %s: score %.4f, velocity rel L2 %.4f, pressure centered rel L2 %.4f -> %s",
             part, checkpoint, summary["score"], summary["velocity_rel_l2"], summary["pressure_centered_rel_l2"],
             stem.with_suffix(".json"))
    return out

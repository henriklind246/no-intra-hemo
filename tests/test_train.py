"""Tests of the models (src/models), the metrics, the epoch sampler and the
training loop (src/train): shapes and periodicity of the MLP; for LinearNO,
chunked evaluation equal to the unchunked model, permutation equivariance,
checkpointing, the zero-initialized conditioning and the early FiLM
geometry injection; metric sums against
direct formulas and chunking; the OneCycle schedule; sampler coverage and
resumption; and end-to-end training, interruption, exact resumption and
evaluation on three synthetic tube cases (one per split part), on the CPU.
"""

import json
import os
import signal

import numpy as np
import pytest

from src.data import cli, splits
from src.data.dataset import EpochSampler, SampledPoints
from src.data.normalize import Normalization, normalization_path
from src.models import PointwiseMLP, TimeConditionedLinearNO, build_model, predict_field
from src.train import cli as train_cli
from src.train import trainer
from src.train.config import TrainConfig
from src.train.metrics import FieldMetrics, metrics_from_sums

from conftest import torch, write_case
from test_geometry import blood_from_points, tube, tube_volume, write_wall_files

pytest.importorskip("scipy")


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    """Three featured tube cases split 1/1/1, with a normalization next to the split."""
    tmp = tmp_path_factory.mktemp("train")
    root = tmp / "aneug_flow"
    pos, faces, nt = tube()
    for i, fs in enumerate([(0.4, 0.6), (0.3, 0.7), (0.55, 0.45)]):
        d = write_case(root / "transient_data", f"stable_{i}", blood_from_points(tube_volume(i), seed=i),
                       flow_split=f"{fs[0]} {fs[1]}\n")
        write_wall_files(d, pos, faces, nt, flow_split=fs)
    assert cli.main(["convert", "--root", str(root), "--all"]) == 0
    assert cli.main(["features", "--root", str(root), "--all"]) == 0
    out = root / "processed_data"
    split_path = splits.create_split(out, val=1 / 3, test=1 / 3, seed=0)
    split = splits.Split.load(split_path)
    assert all(len(split.ids(p)) == 1 for p in ("train", "val", "test"))
    Normalization.from_split(split, out).save(normalization_path(split_path))
    return {"out": out, "split": split_path, "tmp": tmp}


LINEARNO = dict(model="linearno", width=16, depth=2, heads=4, scheduler="onecycle", lr=5e-4)


def tiny(project, name, **kw):
    d = dict(name=name, out_dir=str(project["tmp"] / "runs"), data_root=str(project["out"]),
             split=str(project["split"]), width=16, depth=2, time_harmonics=2, points=32, batch_size=2,
             steps=8, warmup_steps=2, eval_every=100, log_every=2, checkpoint_every=4, num_workers=0,
             device="cpu", eval_chunk=500)
    d.update(kw)
    return TrainConfig(**d)


# --- model ---------------------------------------------------------------------


@pytest.mark.parametrize("fourier", [0, 6])
def test_mlp_shapes_and_periodic_time(fourier):
    torch.manual_seed(0)
    m = PointwiseMLP(width=16, depth=3, time_harmonics=3, fourier_features=fourier)
    x, fs = torch.randn(2, 5, 8), torch.randn(2, 2)
    t = torch.tensor([[0.0, 0.25, 0.5], [0.1, 0.2, 0.3]])
    y = m(x, fs, t)
    assert y.shape == (2, 3, 5, 4)
    assert torch.allclose(m(x, fs, t + 1.0), y, atol=1e-5)  # one full cycle later
    # Each item, and each point, is predicted on its own.
    assert torch.allclose(m(x[1:], fs[1:], t[1:]), y[1:], atol=1e-6)
    assert torch.allclose(m(x[:, 2:3], fs, t), y[:, :, 2:3], atol=1e-6)
    again = build_model("mlp", **m.config)
    again.load_state_dict(m.state_dict())
    assert torch.allclose(again(x, fs, t), y)
    with pytest.raises(ValueError):
        build_model("transformer")


def _randomize_conditioning(m):
    # The conditioning starts at zero; give it weights so the checks below cover it.
    g = torch.Generator().manual_seed(1)
    params = list(m.modulation.parameters()) + list(m.final_modulation.parameters())
    params += list(m.geometry_encoder[-1].parameters())
    params += [b.modulation_table for b in m.blocks]
    for p in params:
        p.data = torch.randn(p.shape, generator=g, dtype=p.dtype) * 0.1


def test_linearno_chunked_field_is_exact_and_equivariant():
    torch.manual_seed(0)
    m = TimeConditionedLinearNO(n_hidden=32, n_layers=3, n_head=4).double().eval()
    _randomize_conditioning(m)
    x, fs = torch.randn(2, 60, 8, dtype=torch.float64), torch.randn(2, 2, dtype=torch.float64)
    t = torch.rand(2, 3, dtype=torch.float64)
    y = m(x, fs, t)
    assert y.shape == (2, 3, 60, 4)
    chunks = predict_field(m, [x[:, :7], x[:, 7:41], x[:, 41:]], fs, t)
    assert torch.allclose(torch.cat(chunks, 2), y, atol=1e-12)  # global attention over all chunks
    perm = torch.randperm(60)
    assert torch.allclose(m(x[:, perm], fs, t), y[:, :, perm], atol=1e-12)
    assert torch.allclose(m(x, fs, t + 1.0), y, atol=1e-12)
    assert not torch.allclose(m(x, fs, t + 0.3), y, atol=1e-6)  # time matters
    assert not torch.allclose(m(x, fs + 0.5, t), y, atol=1e-6)  # flow split matters
    # Attention couples points: dropping points changes the others (unlike the MLP).
    assert not torch.allclose(m(x[:, :30], fs, t), y[:, :, :30], atol=1e-6)


def test_linearno_starts_as_unconditioned_linearno_and_checkpointing_is_exact():
    torch.manual_seed(0)
    m = TimeConditionedLinearNO(n_hidden=16, n_layers=2, n_head=4)
    x, fs, t = torch.randn(1, 20, 8), torch.randn(1, 2), torch.rand(1, 2)
    assert torch.equal(m(x, fs, t), m(x, fs + 1, t + 0.3))  # zero-initialized conditioning
    _randomize_conditioning(m)
    m.train()
    grads = []
    for ckpt in (True, False):
        m.zero_grad()
        m.checkpoint_blocks = ckpt
        m(x, fs, t).square().sum().backward()
        grads.append([p.grad.clone() for p in m.parameters()])
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(*grads))
    again = build_model("linearno", **m.config)
    again.load_state_dict(m.state_dict())
    m.eval(), again.eval()
    assert torch.allclose(again(x, fs, t), m(x, fs, t))


def test_early_film_geometry_injection():
    torch.manual_seed(0)
    m = TimeConditionedLinearNO(n_hidden=16, n_layers=8, n_head=4)
    assert m.inject_after == [0, 1, 2, 3]  # the first half of the blocks
    assert TimeConditionedLinearNO(n_hidden=16, n_layers=6, n_head=4).inject_after == [0, 1, 2]
    assert TimeConditionedLinearNO(n_hidden=16, n_layers=6, n_head=4, injection_blocks=1).inject_after == [0]
    with pytest.raises(ValueError):
        TimeConditionedLinearNO(n_hidden=16, n_layers=4, n_head=4, injection_blocks=4)
    # gamma = 1, beta = 0 at initialization: the same as the model without injection.
    plain = TimeConditionedLinearNO(n_hidden=16, n_layers=8, n_head=4, geometry_injection=False)
    assert plain.geometry_encoder is None and plain.inject_after == []
    missing, unexpected = plain.load_state_dict(m.state_dict(), strict=False)
    assert not missing and all(k.startswith("geometry_encoder") for k in unexpected)
    m.eval(), plain.eval()
    x, fs, t = torch.randn(1, 30, 8), torch.randn(1, 2), torch.rand(1, 2)
    assert torch.equal(m(x, fs, t), plain(x, fs, t))
    # Once trained away from the identity, geometry changes the output, and only after blocks 0-3.
    calls = []
    for i, b in enumerate(m.blocks):
        b.register_forward_hook(lambda mod, args, out, i=i: calls.append((i, args[0].detach().clone(), out)))
    with torch.no_grad():
        m.geometry_encoder[-1].weight.normal_(std=0.1)
    m(x, fs, t)
    assert not torch.allclose(m(x, fs, t), plain(x, fs, t), atol=1e-6)
    calls.clear()
    m(x, fs, t)
    gamma, beta = m._film(x, 2)
    for (i, _, out), (_, nxt, _) in zip(calls, calls[1:]):
        expected = gamma * out + beta if i in m.inject_after else out
        assert torch.allclose(nxt, expected, atol=1e-6), i


def test_onecycle_schedule_follows_the_settings():
    cfg = TrainConfig(scheduler="onecycle", lr=5e-4, steps=200, warmup_fraction=0.05, weight_decay=1e-4)
    m = torch.nn.Linear(2, 2)
    opt, sched = trainer.make_optimizer(m, cfg)
    g = opt.param_groups[0]
    assert g["betas"] == (0.9, 0.999) and g["weight_decay"] == 1e-4
    lrs = []
    for _ in range(cfg.steps):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] == pytest.approx(5e-4 / 25)
    assert int(np.argmax(lrs)) in (9, 10) and max(lrs) == pytest.approx(5e-4, rel=1e-3)  # 5% warmup
    assert lrs[-1] < 1e-6 and all(a >= b for a, b in zip(lrs[10:], lrs[11:]))  # cosine down
    assert opt.param_groups[0]["betas"] == (0.9, 0.999)  # no momentum cycling


# --- metrics -------------------------------------------------------------------


def test_metrics_match_formulas_and_chunking():
    rng = np.random.default_rng(0)
    scale = np.array([0.2, 0.2, 0.2, 400.0])
    target = rng.normal(size=(1, 1, 1000, 4))
    pred = target + rng.normal(scale=0.1, size=target.shape) + np.array([0, 0, 0, 0.5])
    whole = FieldMetrics(["c"], scale)
    whole.update(torch.tensor(pred), torch.tensor(target), torch.tensor([0]), torch.tensor([[3]]))
    chunked = FieldMetrics(["c"], scale)
    for s in (slice(0, 300), slice(300, 1000)):
        chunked.update(torch.tensor(pred[:, :, s]), torch.tensor(target[:, :, s]), torch.tensor([0]),
                       torch.tensor([[3]]))
    (a,), (b,) = whole.fields(), chunked.fields()
    for k in a:
        assert a[k] == pytest.approx(b[k], rel=1e-10) if isinstance(a[k], float) else a[k] == b[k]
    e, t = (pred - target)[0, 0] * scale, target[0, 0] * scale
    assert a["velocity_rel_l2"] == pytest.approx(np.linalg.norm(e[:, :3]) / np.linalg.norm(t[:, :3]))
    assert a["pressure_rel_l2"] == pytest.approx(np.linalg.norm(e[:, 3]) / np.linalg.norm(t[:, 3]))
    ec, tc = e[:, 3] - e[:, 3].mean(), t[:, 3] - t[:, 3].mean()
    assert a["pressure_centered_rel_l2"] == pytest.approx(np.linalg.norm(ec) / np.linalg.norm(tc))
    assert a["pressure_centered_rel_l2"] < a["pressure_rel_l2"]  # the constant offset is removed
    assert a["velocity_mae"] == pytest.approx(np.linalg.norm(e[:, :3], axis=1).mean())
    assert a["normalized_mse"] == pytest.approx(((pred - target) ** 2).mean())
    s = whole.summary()
    assert s["num_fields"] == 1 and s["worst_cases"][0]["case_id"] == "c" and 3 in s["per_snapshot"]
    assert np.isnan(metrics_from_sums(np.zeros(10) + np.r_[1, 0, 0, 0, 0, 0, 0, 0, 0, 0])["velocity_rel_l2"])


# --- sampler -------------------------------------------------------------------


def test_epoch_sampler_covers_epochs_and_resumes():
    keys = []
    for k in EpochSampler(7, seed=1):
        keys.append(k)
        if len(keys) == 21:
            break
    for e in range(3):
        assert sorted(i for ep, i in keys if ep == e) == list(range(7))
    assert [k for ep, k in keys[:7]] != [k for ep, k in keys[7:14]]  # reshuffled
    resumed = []
    for k in EpochSampler(7, seed=1, start=10):
        resumed.append(k)
        if len(resumed) == 11:
            break
    assert resumed == keys[10:]


def test_dataset_accepts_epoch_keys(project):
    split = splits.Split.load(project["split"])
    norm = Normalization.load(normalization_path(project["split"]))
    ds = SampledPoints(split, "train", project["out"], norm, points=16)
    ds.set_epoch(5)
    assert (ds[3]["points"] == ds[(5, 3)]["points"]).all()
    assert not (ds[(6, 3)]["points"] == ds[(5, 3)]["points"]).all()


# --- training ------------------------------------------------------------------


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_train_resume_and_evaluate(project):
    cfg = tiny(project, "basic", steps=6, eval_every=3)
    res = trainer.train(cfg)
    run = cfg.run_dir
    assert res["status"] == "finished" and res["step"] == 6
    for f in ("config.json", "split.json", "normalization.json", "metrics.jsonl", "train.log", "last.pt",
              "best.pt", "status.json"):
        assert (run / f).is_file(), f
    lines = read_jsonl(run / "metrics.jsonl")
    evals = [r for r in lines if "val" in r]
    assert [r["step"] for r in evals] == [3, 6]
    assert all(np.isfinite(r["loss"]) and r["points_per_s"] > 0 for r in lines if "loss" in r)
    assert evals[0]["val"]["num_fields"] == len(cfg.eval_snapshots(4))
    with pytest.raises(FileExistsError):
        trainer.train(cfg)

    res = trainer.resume(run, steps=10)
    assert res["step"] == 10 and TrainConfig.load(run / "config.json").steps == 10

    out = trainer.evaluate_run(run, part="test", device="cpu", num_workers=0)
    assert out["num_fields"] == 4 and out["num_cases"] == 1 and out["step"] in (3, 6, 10)
    assert (run / "eval_test_best.json").is_file()
    assert len((run / "eval_test_best.csv").read_text().splitlines()) == 1 + 4


@pytest.mark.parametrize("variant", ["mlp", "linearno"])
def test_interrupted_run_resumes_exactly(project, monkeypatch, variant):
    extra = LINEARNO if variant == "linearno" else {}
    straight = tiny(project, f"straight_{variant}", **extra)
    assert trainer.train(straight)["step"] == 8

    # Send SIGTERM during step 3: the loop finishes that step, saves and stops.
    real = trainer.loss_fn

    def interrupting(cfg, device):
        f, calls = real(cfg, device), [0]

        def g(pred, target):
            calls[0] += 1
            if calls[0] == 3:
                os.kill(os.getpid(), signal.SIGTERM)
            return f(pred, target)
        return g

    monkeypatch.setattr(trainer, "loss_fn", interrupting)
    cfg = tiny(project, f"interrupted_{variant}", **extra)
    cfg_file = project["tmp"] / f"cfg_{variant}.json"
    args = ["train", "--config", str(cfg_file)]
    cfg_file.write_text(json.dumps(cfg.to_dict()))
    assert train_cli.main(args) == train_cli.EXIT_INTERRUPTED
    status = json.loads((cfg.run_dir / "status.json").read_text())
    assert status["status"] == "interrupted" and status["step"] == 3
    monkeypatch.setattr(trainer, "loss_fn", real)
    assert train_cli.main(["resume", str(cfg.run_dir)]) == 0

    a = trainer.load_checkpoint(straight.run_dir / "last.pt", torch.device("cpu"))["model"]
    b = trainer.load_checkpoint(cfg.run_dir / "last.pt", torch.device("cpu"))["model"]
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_linearno_evaluation_does_not_depend_on_chunking(project):
    cfg = tiny(project, "lno_eval", steps=4, **LINEARNO)
    trainer.train(cfg)
    a = trainer.evaluate_run(cfg.run_dir, checkpoint="last", part="test", device="cpu", num_workers=0, chunk=97)
    b = trainer.evaluate_run(cfg.run_dir, checkpoint="last", part="test", device="cpu", num_workers=0,
                             chunk=100_000)
    for k in ("velocity_rel_l2", "pressure_rel_l2", "pressure_centered_rel_l2", "normalized_mse"):
        assert a[k] == pytest.approx(b[k], rel=1e-4), k
    with pytest.raises(ValueError, match="fixed length"):
        trainer.resume(cfg.run_dir, steps=10)


def test_training_with_worker_processes(project):
    cfg = tiny(project, "workers", steps=4, num_workers=2, eval_every=4)
    assert trainer.train(cfg)["status"] == "finished"


def test_config_flags_override_file(project, tmp_path):
    args = train_cli.build_parser().parse_args(["train", "--width", "64", "--no-amp", "--eval-max-cases", "3"])
    cfg = TrainConfig.from_namespace(args, {"width": 32, "depth": 3})
    assert (cfg.width, cfg.depth, cfg.amp, cfg.eval_max_cases) == (64, 3, False, 3)
    with pytest.raises(ValueError):
        TrainConfig.from_dict({"widht": 3})
    with pytest.raises(ValueError):
        TrainConfig(loss="l1")
    assert TrainConfig(eval_snapshot_stride=10, eval_snapshot_offset=2).eval_snapshots(80)[:2] == [2, 12]

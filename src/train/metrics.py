"""Error metrics of predicted flow fields, in physical units.

``FieldMetrics`` accumulates, for every (case, snapshot) field, sums over its
points (in float64, chunk by chunk, so chunked and whole-field evaluation give
the same numbers) and turns them into per-field metrics:

    velocity_rel_l2           ||u_pred - u|| / ||u||, over the velocity vectors
    pressure_rel_l2           the same for gauge pressure (p - 101325 Pa)
    pressure_centered_rel_l2  the same after removing each field's mean
                              pressure from prediction and truth: only the
                              pressure differences, which carry the physics
    velocity_mae, velocity_rmse   in m/s (norm of the vector error)
    pressure_mae, pressure_centered_rmse   in Pa
    normalized_mse            mean squared error of the normalized outputs
                              (the training loss, for comparison)
    score                     (velocity_rel_l2 + pressure_centered_rel_l2) / 2,
                              used to pick the best checkpoint

All sums weight every point equally (the mesh is finer near the wall, so this
is not a volume average). ``summary`` averages the field metrics over all
fields, and also reports per case (mean over its snapshots) the median and the
worst cases, and per snapshot index the mean over cases.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

SUMS = ("n", "eu2", "u2", "eu_abs", "ep", "ep2", "p", "p2", "ep_abs", "nse")
METRICS = ("velocity_rel_l2", "pressure_rel_l2", "pressure_centered_rel_l2", "velocity_mae", "velocity_rmse",
           "pressure_mae", "pressure_centered_rmse", "normalized_mse", "score")


def field_sums(pred: torch.Tensor, target: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """(..., K, 4) normalized prediction and target -> (..., len(SUMS)) float64 sums over the K points."""
    pred, target = pred.double(), target.double()
    d = pred - target
    e, t = d * scale.double(), target * scale.double()
    eu, u = e[..., :3], t[..., :3]
    ep, p = e[..., 3], t[..., 3]
    n = torch.full(ep.shape[:-1], ep.shape[-1], dtype=torch.float64, device=ep.device)
    return torch.stack([
        n, (eu**2).sum((-1, -2)), (u**2).sum((-1, -2)), eu.norm(dim=-1).sum(-1),
        ep.sum(-1), (ep**2).sum(-1), p.sum(-1), (p**2).sum(-1), ep.abs().sum(-1), (d**2).sum((-1, -2)),
    ], dim=-1)


def metrics_from_sums(s: np.ndarray) -> Dict[str, float]:
    n, eu2, u2, eu_abs, ep, ep2, p, p2, ep_abs, nse = s
    ep_var = max(ep2 - ep**2 / n, 0.0)
    p_var = max(p2 - p**2 / n, 0.0)
    div = lambda a, b: float(np.sqrt(a / b)) if b > 0 else float("nan")
    m = {
        "velocity_rel_l2": div(eu2, u2),
        "pressure_rel_l2": div(ep2, p2),
        "pressure_centered_rel_l2": div(ep_var, p_var),
        "velocity_mae": float(eu_abs / n),
        "velocity_rmse": float(np.sqrt(eu2 / n)),
        "pressure_mae": float(ep_abs / n),
        "pressure_centered_rmse": float(np.sqrt(ep_var / n)),
        "normalized_mse": float(nse / (4 * n)),
    }
    m["score"] = 0.5 * (m["velocity_rel_l2"] + m["pressure_centered_rel_l2"])
    return m


class FieldMetrics:
    def __init__(self, case_ids: Sequence[str], target_scale: Sequence[float]):
        self.case_ids = list(case_ids)
        self.scale = torch.as_tensor(np.asarray(target_scale, dtype=np.float64))
        self._sums: Dict[Tuple[int, int], np.ndarray] = defaultdict(lambda: np.zeros(len(SUMS)))

    def update(self, pred: torch.Tensor, target: torch.Tensor, case: torch.Tensor, snapshots: torch.Tensor) -> None:
        """pred, target (B, s, K, 4) normalized; case (B,); snapshots (B, s)."""
        sums = field_sums(pred, target, self.scale.to(pred.device)).cpu().numpy()
        case, snapshots = case.cpu().numpy(), snapshots.cpu().numpy()
        for b in range(sums.shape[0]):
            for j in range(sums.shape[1]):
                self._sums[(int(case[b]), int(snapshots[b, j]))] += sums[b, j]

    def __len__(self) -> int:
        return len(self._sums)

    def fields(self) -> List[dict]:
        rows = []
        for (c, t), s in sorted(self._sums.items()):
            rows.append({"case_id": self.case_ids[c], "snapshot": t, "num_points": int(s[0]),
                         **metrics_from_sums(s)})
        return rows

    def summary(self, worst: int = 5) -> dict:
        rows = self.fields()
        if not rows:
            return {"num_fields": 0}
        out = {"num_fields": len(rows), "num_cases": len({r["case_id"] for r in rows})}
        out.update({k: float(np.nanmean([r[k] for r in rows])) for k in METRICS})
        by_case = defaultdict(list)
        by_snapshot = defaultdict(list)
        for r in rows:
            by_case[r["case_id"]].append(r)
            by_snapshot[r["snapshot"]].append(r)
        case_score = {c: float(np.nanmean([r["score"] for r in rs])) for c, rs in by_case.items()}
        scores = np.array(list(case_score.values()))
        out["case_score_median"] = float(np.nanmedian(scores))
        out["case_score_p90"] = float(np.nanpercentile(scores, 90))
        out["worst_cases"] = [{"case_id": c, "score": s}
                              for c, s in sorted(case_score.items(), key=lambda kv: -kv[1])[:worst]]
        out["per_snapshot"] = {int(t): {k: float(np.nanmean([r[k] for r in rs])) for k in
                                        ("velocity_rel_l2", "pressure_centered_rel_l2", "score")}
                               for t, rs in sorted(by_snapshot.items())}
        return out

    def write_csv(self, path: Path) -> None:
        rows = self.fields()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["case_id", "snapshot", "num_points", *METRICS])
            w.writeheader()
            w.writerows(rows)

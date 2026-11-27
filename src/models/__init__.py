"""Models mapping a geometry, flow split and time to the flow field.

All models take normalized inputs from ``src.data.dataset`` and return
normalized outputs:
    forward(inputs (B, K, 8), flow_split (B, 2), time (B, s)) -> (B, s, K, 4)

    mlp       PointwiseMLP, the baseline: every point on its own (mlp.py)
    linearno  TimeConditionedLinearNO: linear attention over the whole point
              cloud, conditioned on time and flow split (linearno.py)

Models whose output at a point depends on other points (attention) also
define ``forward_field(chunks, flow_split, time)``, which evaluates a whole
field given as a list of point chunks exactly. ``predict_field`` uses it when
it exists and otherwise predicts each chunk on its own.

``build_model(name, **config)`` creates one; checkpoints store the name and
the model's ``config`` so ``build_model`` can rebuild it.
"""

from __future__ import annotations

from typing import List

from .linearno import TimeConditionedLinearNO
from .mlp import PointwiseMLP

MODELS = {"mlp": PointwiseMLP, "linearno": TimeConditionedLinearNO}


def build_model(name: str, **config):
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; available: {sorted(MODELS)}")
    return MODELS[name](**config)


def predict_field(model, chunks: List, flow_split, time) -> List:
    """Predict one field given as a list of (B, K_i, P) point chunks; returns (B, s, K_i, out) per chunk."""
    if hasattr(model, "forward_field"):
        return model.forward_field(chunks, flow_split, time)
    return [model(c, flow_split, time) for c in chunks]


__all__ = ["MODELS", "PointwiseMLP", "TimeConditionedLinearNO", "build_model", "predict_field"]

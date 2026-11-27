"""Training and evaluation of the flow-field operators (``python -m src.train``).

    config.py   TrainConfig: every setting, from JSON and command-line flags
    trainer.py  train / resume / evaluate_run, checkpoints, run directories
    metrics.py  FieldMetrics: per-field relative L2 and absolute errors
    cli.py      the ``train``, ``resume`` and ``evaluate`` commands
"""

from .config import TrainConfig
from .metrics import FieldMetrics

__all__ = ["FieldMetrics", "TrainConfig"]

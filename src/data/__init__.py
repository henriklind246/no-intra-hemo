"""AneuG-Flow transient data: conversion from the raw download, and access to the result.

Pipeline, one ``aneug`` subcommand per step (see cli.py):
    scan     manifest.py     inventory the raw case directories
    inspect  inspect_raw.py  describe one raw case's files, read-only
    convert  convert.py      blood_data.pt -> coordinates.npy, solution.npy, metadata.json
    verify   convert.py      re-check processed cases against the raw data
    features geometry.py     wall distance and volume geodesics -> geometry_features.npy
    index    processed.py    summarize what has been processed and what failed
    split    splits.py       train/val/test by geometry -> splits/<name>.json
    normalization normalize.py  operator normalization constants for a split
    outlets  outlets.py      measured outlet flow vs flowsplit_ratio.txt order

Supporting modules:
    layout.py      directory layout, file and field names, format version
    raw.py         loading the raw .pt archives and text files
    stats.py       mergeable per-channel statistics stored with each case
    provenance.py  code/environment record stored in each case's metadata

Public API for training code: ``ProcessedCase`` (memory-mapped arrays and
metadata of one case), ``list_cases``, ``pooled_solution_stats`` and
``pooled_geometry_stats``, ``Split`` (a loaded train/val/test split),
``Normalization`` (the operator's input/output scaling), and the datasets
``SampledPoints`` (training) and ``FullField`` (evaluation) in dataset.py.
"""

from .layout import FORMAT_VERSION, SOLUTION_FIELDS, default_root
from .geometry import FEATURE_CHANNELS
from .processed import ProcessedCase, list_cases, pooled_geometry_stats, pooled_solution_stats
from .splits import Split
from .normalize import Normalization
from .dataset import FullField, SampledPoints

__all__ = [
    "FEATURE_CHANNELS",
    "FORMAT_VERSION",
    "FullField",
    "Normalization",
    "SOLUTION_FIELDS",
    "SampledPoints",
    "Split",
    "ProcessedCase",
    "default_root",
    "list_cases",
    "pooled_geometry_stats",
    "pooled_solution_stats",
]

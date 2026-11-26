"""AneuG-Flow transient data: conversion from the raw download, and access to the result.

Pipeline, one ``aneug`` subcommand per step (see cli.py):
    scan     manifest.py     inventory the raw case directories
    inspect  inspect_raw.py  describe one raw case's files, read-only
    convert  convert.py      blood_data.pt -> coordinates.npy, solution.npy, metadata.json
    verify   convert.py      re-check processed cases against the raw data
    index    processed.py    summarize what has been processed and what failed

Supporting modules:
    layout.py      directory layout, file and field names, format version
    raw.py         loading the raw .pt archives and text files
    stats.py       mergeable per-channel statistics stored with each case
    provenance.py  code/environment record stored in each case's metadata

Public API for training code: ``ProcessedCase`` (memory-mapped arrays and
metadata of one case), ``list_cases`` and ``pooled_solution_stats``.
"""

from .layout import FORMAT_VERSION, SOLUTION_FIELDS, default_root
from .processed import ProcessedCase, list_cases, pooled_solution_stats

__all__ = [
    "FORMAT_VERSION",
    "SOLUTION_FIELDS",
    "ProcessedCase",
    "default_root",
    "list_cases",
    "pooled_solution_stats",
]

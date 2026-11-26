"""Neural operators for intracranial aneurysm hemodynamics (imported as ``src``).

Learns transient blood-flow velocity and pressure fields in intracranial
aneurysm geometries from the AneuG-Flow CFD dataset.

Subpackages:
    data  Conversion of the raw AneuG-Flow transient cases into a compact,
          memory-mapped layout, access to the processed cases, and the
          ``aneug`` command line tool.

Model and training code are not written yet.
"""

__version__ = "0.1.0"

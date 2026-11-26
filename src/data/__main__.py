"""Entry point for ``python -m no_intra_hemo.data``; identical to the ``aneug`` command.

Lets the CLI in cli.py run without installing the package, as long as the
repo root is the working directory or on PYTHONPATH. This is how
scripts/msi/run.sh invokes it on MSI.
"""

import sys

from .cli import main

sys.exit(main())

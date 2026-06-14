"""KempnerPulse entry point: parse config, run the pipeline.

    python -m kempnerpulse [options]
    kempnerpulse [options]            (installed console script)
"""
from __future__ import annotations

import sys
from typing import List, Optional

from .config import build_config
from .lifecycle import run


def main(argv: Optional[List[str]] = None) -> int:
    return run(build_config(argv))


if __name__ == "__main__":
    sys.exit(main())

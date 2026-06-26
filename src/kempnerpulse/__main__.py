"""KempnerPulse entry point for the V100 custom branch.

The ``v100-custom`` branch intentionally runs the same implementation as the
single-file ``kempner_pulse_nvlink_fast_poll_fixed.py`` script. Keeping the
console entry point delegated here prevents the package TUI from diverging from
the known-good single-file behavior.
"""
from __future__ import annotations

import sys
from typing import List, Optional

from . import _v100_single_file


def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        return _v100_single_file.main()

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *argv]
        return _v100_single_file.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    sys.exit(main())

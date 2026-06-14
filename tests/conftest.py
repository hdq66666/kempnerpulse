"""Make both the legacy module and the src/ package importable in tests.

The legacy ``kempner_pulse`` lives at the repo root; the layered ``kempnerpulse``
package lives under ``src/``. Adding both to ``sys.path`` lets the unit tests run
without an editable install of the src-layout package.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_ROOT, "src"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

"""Cross-cutting tier — GPU visibility selection.

Resolves *which* GPU ids the dashboard should monitor, by precedence, from an
explicit ``--gpus`` list, ``--show-all``, the SLURM/CUDA visibility environment,
and finally the set of accessible ids. The result is always intersected
("clamped") with the accessible set so a selection can never widen visibility
beyond what the process can actually reach.

This module reads ``os.environ`` and parses id strings; it does no I/O and never
shells out. Runtime dependencies are the standard library only.
"""
from __future__ import annotations

import os
import re
from typing import Optional, Set, Tuple

# Visibility environment variables, in resolution order (first non-empty whose
# parsed ids survive clamping wins). Mirrors the documented selection order.
_ENV_CANDIDATES = (
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "SLURM_STEP_GPUS",
    "SLURM_JOB_GPUS",
)


class GPUSelector:
    """Resolve the set of GPU ids to monitor by precedence.

    Args:
        explicit: the raw ``--gpus`` value, or ``None`` if unset.
        disable_auto: the ``--show-all`` flag — ignore the environment and use
            the full accessible set.
        accessible: ids the process can actually reach (from nvidia-smi /
            dcgmi). ``None`` means "unknown", in which case no clamping is
            applied and selections pass through unfiltered.
    """

    def __init__(
        self,
        explicit: Optional[str],
        disable_auto: bool = False,
        accessible: Optional[Set[str]] = None,
    ) -> None:
        self.explicit = explicit
        self.disable_auto = disable_auto
        self.accessible = accessible
        self.allowed: Optional[Set[str]] = None
        self.reason: str = "all"
        self.source_value: Optional[str] = None

    def _clamp(self, ids: Optional[Set[str]]) -> Optional[Set[str]]:
        """Intersect ``ids`` with the accessible set (no-op if either is None)."""
        if ids is None or self.accessible is None:
            return ids
        return ids & self.accessible

    def resolve(self) -> Tuple[Optional[Set[str]], str, Optional[str]]:
        """Return ``(allowed_ids, reason, source_value)``.

        ``allowed_ids`` is the clamped id set to monitor (``None`` = "all
        accessible, unfiltered"). ``reason`` names the precedence rule that
        decided it (``"--gpus"``, an env var name, or ``"all"``).
        ``source_value`` is the raw string that rule used, or ``None``.
        """
        if self.explicit:
            ids = self._clamp(self._parse_gpu_list(self.explicit))
            self.allowed = ids
            self.reason = "--gpus"
            self.source_value = self.explicit
            return self.allowed, self.reason, self.source_value

        if self.disable_auto:
            self.allowed = self._clamp(self.accessible)
            self.reason = "all"
            self.source_value = None
            return self.allowed, self.reason, self.source_value

        for key in _ENV_CANDIDATES:
            raw = os.environ.get(key, "").strip()
            if not raw:
                continue
            ids = self._clamp(self._parse_gpu_list(raw))
            if ids:
                self.allowed = ids
                self.reason = key
                self.source_value = raw
                return self.allowed, self.reason, self.source_value

        self.allowed = self._clamp(self.accessible)
        self.reason = "all"
        self.source_value = None
        return self.allowed, self.reason, self.source_value

    @staticmethod
    def _parse_gpu_list(raw: str) -> Set[str]:
        """Parse a GPU-id spec into a set of numeric id strings.

        Accepts comma lists, ``a-b`` ranges, bracketed hostlist-style ranges
        (``node[0-3]``), and suffixed tokens (``gpu2``, ``0000:.../gpu3``). The
        sentinels ``all`` / ``none`` / ``void`` resolve to the empty set.
        """
        raw = raw.strip()
        if not raw or raw.lower() in {"all", "none", "void"}:
            return set()

        ids: Set[str] = set()
        for part in raw.split(","):
            token = part.strip()
            if not token:
                continue

            bracket = re.match(r"^[^\[]*\[(.+)\]$", token)
            if bracket:
                ids |= GPUSelector._expand_ranges(bracket.group(1))
                continue

            if re.fullmatch(r"\d+(?:-\d+)?", token):
                ids |= GPUSelector._expand_ranges(token)
                continue

            suffix_num = re.search(r"(?:^|[:/])(?:gpu)?(\d+)$", token, flags=re.IGNORECASE)
            if suffix_num:
                ids.add(suffix_num.group(1))
                continue

            embedded_nums = re.findall(r"\d+", token)
            if embedded_nums and token.lower().startswith("gpu"):
                ids.add(embedded_nums[-1])
                continue

        return ids

    @staticmethod
    def _expand_ranges(raw: str) -> Set[str]:
        """Expand a comma-list of ``a-b`` ranges and bare ints into id strings."""
        out: Set[str] = set()
        for chunk in raw.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            m = re.fullmatch(r"(\d+)-(\d+)", chunk)
            if m:
                start, end = int(m.group(1)), int(m.group(2))
                low, high = min(start, end), max(start, end)
                out |= {str(i) for i in range(low, high + 1)}
            elif chunk.isdigit():
                out.add(chunk)
        return out

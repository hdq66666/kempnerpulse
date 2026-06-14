"""Temporary compatibility shims.

Thin, internal bridges that let the legacy ``kempner_pulse`` module consume the
layered components while the codebase transitions to them. These are
transitional by design and will be removed once the transition is complete;
they are not part of the public API and are intentionally not exported from the
top-level package.
"""
from .raw_to_sample import (
    DcgmStreamReader,
    Sample,
    load_dcgm_direct,
    parse_dcgm_dmon,
    parse_prometheus_text,
    records_to_dcgm_sample,
    records_to_sample,
    sample_to_records,
)

__all__ = [
    "Sample",
    "DcgmStreamReader",
    "parse_dcgm_dmon",
    "parse_prometheus_text",
    "load_dcgm_direct",
    "records_to_sample",
    "records_to_dcgm_sample",
    "sample_to_records",
]

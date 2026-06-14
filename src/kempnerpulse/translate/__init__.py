"""KempnerPulse Layer 2 — Translate.

Maps Layer-1 ``RawRecord``s (source vocabulary, raw values) to
``CanonicalRecord``s (one stable internal vocabulary). This package currently
defines the canonical schema — the inter-layer contract that Layers 3 and 4
depend on. The translator that produces canonical records (mapping tables, unit
normalization, missing-value policy, counter differencing) lands on top of it.
"""
from .context import SourceContext, make_source_context
from .schema import (
    POINT_WINDOW_MAX_MICROSECONDS,
    SCHEMA_VERSION,
    AggregationMode,
    CanonicalRecord,
    Provenance,
    TranslateError,
    canonical_field_names,
)
from .translator import Translator, make_translator

__all__ = [
    "CanonicalRecord",
    "AggregationMode",
    "Provenance",
    "TranslateError",
    "SCHEMA_VERSION",
    "POINT_WINDOW_MAX_MICROSECONDS",
    "canonical_field_names",
    "Translator",
    "make_translator",
    "SourceContext",
    "make_source_context",
]

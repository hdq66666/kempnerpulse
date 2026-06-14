"""KempnerPulse — GPU monitoring on the NVIDIA DCGM backend.

The package is organized as four data-flow layers — Read -> Translate ->
Compute -> Present — over a cross-cutting tier (configuration, observability,
lifecycle, errors). Data flows top to bottom: each layer depends only on the
ones above it.
"""

__all__: list[str] = []

"""Guard tests for the workload-status taxonomy and the status-column width.

The dashboard renders the workload status in a fixed-width column
(``STATUS_DISPLAY_WIDTH``, derived from ``WORKLOAD_STATUS_LABELS``). These tests
fail if a future change would silently break that layout:

  * a new status label longer than today's longest, or
  * a status returned by ``derive_real_util`` that isn't registered in
    ``WORKLOAD_STATUS_LABELS`` (the list the width is derived from).

Run with ``pytest`` or directly: ``python tests/test_status_labels.py``.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kempner_pulse as kp  # noqa: E402

# Longest status label today is "tensor-heavy compute" (20 chars). If a future
# status exceeds this, bump this constant *deliberately* after re-checking the
# status-column width (STATUS_DISPLAY_WIDTH) and the card detail-column widths.
MAX_STATUS_LEN_TODAY = 20


def _classifier_status_labels():
    """Status strings ``derive_real_util`` can return, parsed from the source."""
    src = open(kp.__file__, encoding="utf-8").read()
    return set(re.findall(r'return\s+real_util\s*,\s*"[^"]*"\s*,\s*"([^"]+)"', src))


def test_no_status_label_longer_than_current_max():
    longest = max(kp.WORKLOAD_STATUS_LABELS, key=len)
    assert len(longest) <= MAX_STATUS_LEN_TODAY, (
        f"New status {longest!r} is {len(longest)} chars, exceeding the current max "
        f"of {MAX_STATUS_LEN_TODAY}. Bump MAX_STATUS_LEN_TODAY and re-check the status "
        f"column width (STATUS_DISPLAY_WIDTH) and CARD_DETAIL_RIGHT_VALUE_MIN."
    )


def test_classifier_statuses_are_registered():
    returned = _classifier_status_labels()
    registered = set(kp.WORKLOAD_STATUS_LABELS)
    assert returned and returned <= registered, (
        "derive_real_util returns status label(s) missing from WORKLOAD_STATUS_LABELS: "
        f"{sorted(returned - registered)}. Register them so the derived status-column "
        "width stays correct."
    )


def test_display_width_covers_longest_status():
    assert kp.STATUS_DISPLAY_WIDTH >= max(len(s) for s in kp.WORKLOAD_STATUS_LABELS), (
        "STATUS_DISPLAY_WIDTH must be at least the longest status label."
    )


# Pinned longest health-badge label today: "WARN"/"CRIT" (4). The card pads the badge
# to a fixed width so it never wraps; a longer label would change that width.
MAX_HEALTH_LEN_TODAY = 4


def _health_labels_returned():
    src = open(kp.__file__, encoding="utf-8").read()
    start = src.index("def health_from_metrics")
    body = src[start: src.index("\ndef ", start)]
    return set(re.findall(r'return\s+"([A-Za-z]+)"\s*,', body))


def test_no_health_label_longer_than_current_max():
    longest = max(kp.HEALTH_LABELS, key=len)
    assert len(longest) <= MAX_HEALTH_LEN_TODAY, (
        f"New health label {longest!r} ({len(longest)} chars) exceeds the current max "
        f"({MAX_HEALTH_LEN_TODAY}); bump MAX_HEALTH_LEN_TODAY and re-check the card "
        f"health-badge width (HEALTH_LABEL_WIDTH)."
    )


def test_health_labels_are_registered():
    returned = _health_labels_returned()
    assert returned and returned <= set(kp.HEALTH_LABELS), (
        "health_from_metrics returns label(s) missing from HEALTH_LABELS: "
        f"{sorted(returned - set(kp.HEALTH_LABELS))}."
    )


if __name__ == "__main__":
    test_no_status_label_longer_than_current_max()
    test_classifier_statuses_are_registered()
    test_display_width_covers_longest_status()
    test_no_health_label_longer_than_current_max()
    test_health_labels_are_registered()
    print(
        "OK: guard tests passed "
        f"(status max = {max(map(len, kp.WORKLOAD_STATUS_LABELS))}, "
        f"STATUS_DISPLAY_WIDTH = {kp.STATUS_DISPLAY_WIDTH}, "
        f"health max = {max(map(len, kp.HEALTH_LABELS))})"
    )

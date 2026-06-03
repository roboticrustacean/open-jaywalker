"""Pure plan-summary helper shared by the add-on panel and headless console.

Reduces the classifier report + build plan to the few fields a human needs to
decide whether to build. No bpy, no I/O.
"""

from __future__ import annotations

from phase3_classifier.classifier import CORE_TARGETS


def _aggregate_missing_by_target(report_characters: list) -> list:
    """Count, per target, how many crowd characters are missing it (desc by count)."""
    counts = {}
    for character in report_characters:
        for target in character.get("missing_targets", []):
            counts[target] = counts.get(target, 0) + 1
    return [
        {"target": target, "count": count}
        for target, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def summarize_plan(classifier_report: dict, build_plan: dict) -> dict:
    """Summarize a classifier report + build plan for display.

    Returns: recommended_armature, total (28), mapped, missing_targets (list),
    is_crowd, character_count.
    """
    total = len(CORE_TARGETS)
    missing = list(classifier_report.get("missing_targets", []))
    characters = build_plan.get("characters") or []
    missing_by_target = _aggregate_missing_by_target(classifier_report.get("characters") or [])
    return {
        "missing_by_target": missing_by_target,
        "recommended_armature": classifier_report.get("recommended_primary_armature", ""),
        "total": total,
        "mapped": total - len(missing),
        "missing_targets": missing,
        "is_crowd": bool(characters),
        "character_count": len(characters),
        "review_flags": list(classifier_report.get("review_flags", [])),
        "character_ids": [c.get("character_id", "") for c in characters],
    }

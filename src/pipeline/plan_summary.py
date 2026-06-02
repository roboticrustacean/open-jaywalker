"""Pure plan-summary helper shared by the add-on panel and headless console.

Reduces the classifier report + build plan to the few fields a human needs to
decide whether to build. No bpy, no I/O.
"""

from __future__ import annotations

from phase3_classifier.classifier import CORE_TARGETS


def summarize_plan(classifier_report: dict, build_plan: dict) -> dict:
    """Summarize a classifier report + build plan for display.

    Returns: recommended_armature, total (28), mapped, missing_targets (list),
    is_crowd, character_count.
    """
    total = len(CORE_TARGETS)
    missing = list(classifier_report.get("missing_targets", []))
    characters = build_plan.get("characters") or []
    return {
        "recommended_armature": classifier_report.get("recommended_primary_armature", ""),
        "total": total,
        "mapped": total - len(missing),
        "missing_targets": missing,
        "is_crowd": bool(characters),
        "character_count": len(characters),
        "review_flags": list(classifier_report.get("review_flags", [])),
        "character_ids": [c.get("character_id", "") for c in characters],
    }

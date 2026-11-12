"""Aggregate persisted, validated batch reports without inventing coverage."""

from collections import Counter


def summarize(manifests: list[dict]) -> dict:
    reports = [item for item in manifests if item["kind"] == "report" and "coverage" in item["payload"]]
    latest = {}
    for item in reports:
        batch_id = item["payload"]["batch_id"]
        previous = latest.get(batch_id)
        if previous is None or item["created_at"] > previous["created_at"]:
            latest[batch_id] = item
    batches = list(latest.values())
    return {
        "artifact_count": len(manifests), "kinds": dict(Counter(item["kind"] for item in manifests)),
        "artifact_ids": [item["artifact_id"] for item in manifests],
        "coverage": batches[0]["payload"]["coverage"] if len(batches) == 1 else None,
        "batches": [{"artifact_id": item["artifact_id"], **item["payload"]} for item in batches],
        "note": "Each batch retains its own declared candidate denominator; fractions are not added across batches.",
    }

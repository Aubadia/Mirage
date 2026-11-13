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


def summarize_properties(report):
    """Observed generation funnels; no ground-truth rates are inferred."""
    rows=report.get('records',[])
    generated=sum(bool(r.get('generated')) for r in rows)
    valid=sum(bool(r.get('contract_valid')) for r in rows)
    executed=sum(bool(r.get('executable') and r.get('api_executed')) for r in rows)
    detected=[r for r in rows if r.get('verdict')=='suspected_failure']
    return dict(generated=generated,contract_valid=valid,executable=executed,
                valid_test_rate=valid/generated if generated else None,
                executable_test_rate=executed/valid if valid else None,
                suspected_failures=len(detected),
                time_to_first_suspected_failure=min((r['finished_seconds'] for r in detected),default=None),
                false_positive_rate=None,correctness_violation_detection_rate=None,
                truth_status='Requires independent adjudication',stop_reason=report.get('stop_reason'))

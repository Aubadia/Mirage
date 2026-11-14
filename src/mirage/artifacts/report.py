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


def verified_search_units(row, basis='semantic'):
    """Feedback units require runtime validity, including local preservation."""
    import json
    from mirage.contracts.capabilities import SEMANTIC_COVERAGE_VERSION
    if (not row.get('executable') or not row.get('api_executed')
            or not row.get('inspection', {}).get('valid')
            or row.get('mutation_obligations_valid') is False):
        return set()
    if basis == 'transition':
        signatures = [{'api': row['api'], 'transition': row.get('transition_signature', [])}]
    elif basis == 'semantic':
        coverage = row['inspection'].get('semantic_coverage', {})
        if not coverage.get('eligible') or coverage.get('version') != SEMANTIC_COVERAGE_VERSION:
            return set()
        signatures = coverage.get('signatures', [])
    else:
        raise ValueError('Unknown coverage basis')
    return {json.dumps(s, sort_keys=True, separators=(',', ':')) for s in signatures}


def summarize_search(report):
    """Comparable observed metrics for every treatment, not theoretical coverage."""
    rows = report.get('records', [])
    valid = [r for r in rows if verified_search_units(r)]
    semantic = {unit for r in valid for unit in verified_search_units(r)}
    transitions = {unit for r in valid for unit in verified_search_units(r, 'transition')}
    applicable = [r for r in valid if r.get('relation') and r.get('verdict') in ('pass', 'suspected_failure')]
    pairs = {(r['api'], r['relation']) for r in applicable}
    state_relations = {(unit, r['relation']) for r in applicable for unit in verified_search_units(r)}
    elapsed = report.get('elapsed_seconds', 0)
    usage = [r.get('generation_usage') or {} for r in rows]
    known_tokens = sum(u.get('input_tokens') or 0 for u in usage) + sum(u.get('output_tokens') or 0 for u in usage)
    tokens_complete = bool(rows) and all(u.get('input_tokens') is not None and u.get('output_tokens') is not None for u in usage)
    return {'attempts':len(rows), 'runtime_valid':len(valid),
            'runtime_valid_rate':len(valid)/len(rows) if rows else None,
            'verified_apis':sorted({r['api'] for r in valid}),
            'semantic_units':len(semantic), 'transition_units':len(transitions),
            'applicable_comparisons':len(applicable), 'api_mr_units':len(pairs),
            'state_mr_units':len(state_relations),
            'mr_applicability_rate':len(applicable)/len(valid) if valid else None,
            'coverage_only_cases':sum(r.get('verdict') not in ('pass','suspected_failure') for r in valid),
            'verdicts':dict(Counter(str(r.get('verdict')) for r in rows)),
            'blocking_stages':dict(Counter(r.get('first_blocking_stage','none') for r in rows)),
            'local_attempts':sum(r.get('generation_mode') == 'local' for r in rows),
            'elapsed_seconds':elapsed, 'metadata_seconds':report.get('metadata_seconds'),
            'known_tokens':known_tokens, 'token_usage_complete':tokens_complete,
            'semantic_units_per_second':len(semantic)/elapsed if elapsed else None,
            'semantic_units_per_1000_tokens':1000*len(semantic)/known_tokens if tokens_complete and known_tokens else None,
            'candidate_denominator':'per-decision statically plannable candidates; no theoretical coverage fraction',
            'confirmed_defects':None, 'stop_reason':report.get('stop_reason')}

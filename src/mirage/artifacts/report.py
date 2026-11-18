"""Aggregate persisted, validated batch reports without inventing coverage."""

from collections import Counter


def summarize_training_candidates(canonicals, cases):
    """Counts have explicit units; admission never supplies verified coverage."""
    candidates = [c for row in canonicals for c in row.get("candidates", ())]
    executed = [c for c in cases if c.get("status") == "executed"]
    return {
        "basis": "finite candidate instances per canonical; attempts per sampled path",
        "registry": dict(Counter(c["status"] for c in candidates)),
        "reasons": dict(Counter(reason for c in candidates for reason in c["reasons"])),
        "registered_instances": len(candidates),
        "sampled_candidate_instances": len({(c["id"].rsplit("-t", 1)[0], c["candidate_id"])
                                            for c in cases if c.get("candidate_id")}),
        "sampled_path_attempts": len(cases), "executed_path_attempts": len(executed),
        "attained_path_attempts": sum(c.get("verified") is True for c in executed),
        "mr_applicable_path_attempts": sum(c.get("verdict") in ("pass", "suspected_failure") for c in executed),
        "verdicts": dict(Counter(c.get("verdict") or "unassessed" for c in executed)),
        "unmet_capability_requirements": sorted({key for row in canonicals
                                                 for key in row.get("unmet_capability_requirements", ())}),
    }


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


def verified_optimizer_units(row, *, member='source'):
    """Keep optimizer-chain evidence versioned and separate from operand coverage."""
    import json
    from mirage.contracts.capabilities import OPTIMIZER_COVERAGE_VERSION
    if (not row.get('executable') or not row.get('api_executed')
            or row.get('mutation_obligations_valid') is False):
        return set()
    if member == 'source':
        inspection = row.get('inspection', {})
        if not inspection.get('valid'):
            return set()
        coverage = inspection.get('optimizer_coverage', {})
    elif member == 'followup':
        coverage = row.get('followup_optimizer_coverage', {})
    else:
        raise ValueError('Unknown execution member')
    if not coverage.get('eligible') or coverage.get('version') != OPTIMIZER_COVERAGE_VERSION:
        return set()
    return {json.dumps(unit, sort_keys=True, separators=(',', ':'))
            for unit in coverage.get('signatures', [])
            if unit.get('version') == OPTIMIZER_COVERAGE_VERSION}


def verified_search_units(row, basis='semantic', *, include_optimizer=True):
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
        signatures = [unit for unit in coverage.get('signatures', [])
                      if unit.get('version', SEMANTIC_COVERAGE_VERSION) == SEMANTIC_COVERAGE_VERSION]
    else:
        raise ValueError('Unknown coverage basis')
    units = {json.dumps(s, sort_keys=True, separators=(',', ':')) for s in signatures}
    if basis == 'semantic' and include_optimizer:
        units.update(verified_optimizer_units(row))
    return units


def summarize_optimizer_search(rows, elapsed):
    """An absent observer is unknown; followup attainment never credits the source."""
    from mirage.contracts.capabilities import OPTIMIZER_COVERAGE_VERSION
    measured = [r for r in rows if r.get('inspection', {}).get('optimizer_coverage', {}).get('version')
                == OPTIMIZER_COVERAGE_VERSION]
    known_counts = [r['inspection']['optimizer_coverage']['target_update_count'] for r in measured
                    if r['inspection']['optimizer_coverage'].get('count_verified') is True
                    and type(r['inspection']['optimizer_coverage'].get('target_update_count')) is int]
    linked = [r for r in measured if verified_optimizer_units(r)]
    units = {u for r in linked for u in verified_optimizer_units(r)}
    followup = [r for r in rows if verified_optimizer_units(r, member='followup')]
    followup_units = {u for r in followup for u in verified_optimizer_units(r, member='followup')}
    goals = [r for r in rows if any(p.get('dimension') in ('gradient_representation', 'optimizer')
                                   for p in r.get('configuration', {}).get('predicates', []))]
    gradient_goals = [r for r in goals if any(p.get('dimension') == 'gradient_representation'
                                             for p in r['configuration']['predicates'])]
    def attained(row):
        return bool(verified_optimizer_units(row)) and row['inspection'].get('target_state_reached') is True
    selected = [r for r in rows if r.get('relation') == 'optimizer_gradient_representation']
    verdicts = Counter(str(r.get('mr_verdict', r.get('verdict')))
                       if r.get('inspection', {}).get('valid') and r.get('mutation_obligations_valid') is not False
                       else 'invalid' for r in selected)
    amounts = [r.get('max_abs_parameter_update', {}).get('source', {}) for r in selected]
    measured_amounts = [a for a in amounts if a and all(type(v) in (int, float) for v in a.values())]
    return {'version': OPTIMIZER_COVERAGE_VERSION, 'measured_source_cases': len(measured),
            'unmeasured_source_cases': len(rows) - len(measured),
            'measured_count_unknown_cases': len(measured) - len(known_counts),
            'measured_verified_zero_count_cases': sum(count == 0 for count in known_counts),
            'measured_verified_nonzero_count_cases': sum(count > 0 for count in known_counts),
            'source_linked_cases': len(linked), 'source_units': len(units),
            'source_linked_by_origin': dict(Counter('planned' if r.get('state_plan') else
                                                   'mutation' if r.get('parent_seed_id') else 'fresh'
                                                   for r in linked)),
            'followup_linked_cases': len(followup), 'followup_units': len(followup_units),
            'goal_attempts': len(goals), 'goal_attained_cases': sum(attained(r) for r in goals),
            'gradient_goal_attempts': len(gradient_goals),
            'gradient_update_conjunction_attained_cases': sum(attained(r) for r in gradient_goals),
            'mr_selected_cases': len(selected), 'mr_verdicts': dict(verdicts),
            'mr_applicable_cases': verdicts['pass'] + verdicts['suspected_failure'],
            'source_update_amount_observed_cases': len(measured_amounts),
            'source_nonzero_parameter_update_cases': sum(any(v > 0 for v in a.values()) for a in measured_amounts),
            'source_unchanged_parameter_cases': sum(all(v == 0 for v in a.values()) for a in measured_amounts),
            'source_units_per_second': len(units) / elapsed if elapsed else None,
            'limitations': dict(Counter(str(reason) for r in measured for reason in
                                       r['inspection']['optimizer_coverage'].get('limitations', []))),
            'goal_denominator': 'all attempts requesting optimizer or gradient-representation predicates',
            'coverage_denominator': None,
            'scope': 'bounded target-linked native updates; source and followup counted separately; no theoretical coverage fraction'}


def verified_representation_state_units(row, *, member='source'):
    """Project verified unary coverage; operand layout and training state stay separate."""
    import json
    from mirage.contracts.capabilities import SEMANTIC_COVERAGE_VERSION
    if (not row.get('executable') or not row.get('api_executed')
            or row.get('mutation_obligations_valid') is False):
        return set(), set()
    if member not in ('source', 'followup'):
        raise ValueError('Unknown execution member')
    inspection = row.get('inspection' if member == 'source' else 'followup_inspection', {})
    coverage = inspection.get('semantic_coverage', {})
    if (inspection.get('valid') is not True or coverage.get('eligible') is not True
            or coverage.get('version') != SEMANTIC_COVERAGE_VERSION):
        return set(), set()
    representations, states = set(), set()
    for unit in coverage.get('signatures', []):
        if unit.get('unit_kind', 'unary') != 'unary':
            continue
        base = {k: unit[k] for k in ('api', 'checkpoint', 'operand_path') if k in unit}
        if 'api' not in base:
            continue
        if isinstance(unit.get('representation'), dict):
            representations.add(json.dumps(dict(base, representation=unit['representation']), sort_keys=True))
        if isinstance(unit.get('state'), dict):
            states.add(json.dumps(dict(base, state=unit['state'],
                       execution_mode=unit.get('execution_mode'), environment=unit.get('environment')), sort_keys=True))
    return representations, states


def summarize_exploration_oracle(report, elapsed, known_tokens, tokens_complete):
    """Bounded candidate funnel and effective oracle judgments, with explicit units."""
    rows = report.get('records', [])
    coverage = {}
    all_representations, all_states = set(), set()
    for member in ('source', 'followup'):
        representations, states = set(), set()
        measured = 0
        for row in rows:
            r, s = verified_representation_state_units(row, member=member)
            representations.update(r); states.update(s)
            measured += bool(r or s)
        coverage[member] = {'representation_units': len(representations), 'training_state_units': len(states),
                            'measured_cases': measured, 'unmeasured_cases': len(rows)-measured}
        all_representations.update(representations); all_states.update(states)
    traversal = report.get('representation_traversal', {})
    sources = traversal.get('sources', [])
    candidates = [c for source in sources for c in source.get('candidates', [])]
    statuses = Counter(c.get('status', 'unknown') for c in candidates)
    eligible = [c for c in candidates if c.get('status') in ('pending', 'dispatched', 'reached', 'not_verified', 'previously_reached')]
    dispatched = sum(c.get('status') in ('dispatched', 'reached', 'not_verified') for c in eligible)
    effective_rows = [r for r in rows if r.get('inspection', {}).get('valid')
                      and any(a.get('status') in ('agreement', 'violation') for a in r.get('gradient_audits', {}).values())]
    valid = [r for r in rows if r.get('inspection', {}).get('valid')]
    effective_members = sum(a.get('status') in ('agreement', 'violation')
                            for r in effective_rows for a in r.get('gradient_audits', {}).values())
    suspects = [r for r in rows if r.get('verdict') == 'suspected_failure']
    return {'version': 'exploration-oracle-metrics-1', 'coverage_by_member': coverage,
            'representation_units_union': len(all_representations), 'training_state_units_union': len(all_states),
            'representation_units_per_second': len(all_representations)/elapsed if elapsed else None,
            'representation_units_per_1000_tokens': 1000*len(all_representations)/known_tokens if tokens_complete and known_tokens else None,
            'coverage_scope': 'Verified unary projections, source/followup union; no theoretical coverage denominator. Missing legacy followup inspection is unmeasured.',
            'traversal': {'enabled': traversal.get('enabled', False), 'measured_inventory': 'sources' in traversal,
                          'sources': len(sources), 'candidate_count': len(candidates), 'statuses': dict(statuses),
                          'eligible_candidates': len(eligible), 'dispatched_candidates': dispatched,
                          'ineligible_statuses': dict(Counter(c.get('eligibility', {}).get('status', 'unknown')
                                                              for c in candidates if c.get('status') == 'ineligible')),
                          'ineligible_reasons': dict(Counter(c.get('eligibility', {}).get('reason', 'unspecified')
                                                             for c in candidates if c.get('status') == 'ineligible')),
                          'reached_candidates': statuses['reached'], 'pending_candidates': statuses['pending'],
                          'previously_reached_candidates': statuses['previously_reached'],
                          'tested_candidates': statuses['reached'] + statuses['previously_reached'],
                          'dispatch_fraction': dispatched/len(eligible) if eligible else None,
                          'attainment_fraction': statuses['reached']/dispatched if dispatched else None,
                          'candidate_denominator': 'Observed source-specific single-action candidates; repeated sources remain separate; no theoretical total'},
            'effective_numerical_cases': len(effective_rows), 'effective_numerical_members': effective_members,
            'effective_case_fraction': len(effective_rows)/len(valid) if valid else None,
            'effective_case_denominator': 'Runtime-valid source cases; at least one source/followup agreement or violation',
            'effective_members_per_second': effective_members/elapsed if elapsed else None,
            'audit_schedule_reasons': dict(Counter(r.get('audit_schedule', {}).get('reason', 'unrecorded') for r in rows)),
            'audit_admission_statuses': dict(Counter(a.get('status', 'unknown') for r in rows
                                                   for a in r.get('audit_admissions', {}).values())),
            'audit_admission_reasons': dict(Counter(a.get('reason', 'unspecified') for r in rows
                                                  for a in r.get('audit_admissions', {}).values())),
            'audit_admission_routes': dict(Counter(a.get('route', 'none') for r in rows
                                                 for a in r.get('audit_admissions', {}).values())),
            'suspected_cases': len(suspects),
            'automatically_reproduced_cases': sum(any(p.get('replay_valid') for p in r.get('automatic_replays', [])) for r in suspects),
            'time_to_first_suspect': min((r['finished_seconds'] for r in suspects if 'finished_seconds' in r), default=None),
            'confirmed_distinct_defects': None}


def summarize_search(report):
    """Comparable observed metrics for every treatment, not theoretical coverage."""
    import json
    from mirage.contracts.capabilities import SEMANTIC_COVERAGE_VERSION
    rows = report.get('records', [])
    valid = [r for r in rows if verified_search_units(r)]
    semantic = {unit for r in valid for unit in verified_search_units(r, include_optimizer=False)}
    unit_kinds = Counter(json.loads(unit).get('unit_kind', 'unary') for unit in semantic)
    measurement_counts = Counter()
    for row in valid:
        measurement_counts.update(row['inspection']['semantic_coverage'].get('measurement_counts', {}))
    relational_goals = [row for row in rows if any(
        p.get('dimension') == 'tensor_relation' for p in row.get('configuration', {}).get('predicates', []))]
    relational_attained = sum(bool(verified_search_units(row)) and row['inspection'].get('target_state_reached') is True
                              for row in relational_goals)
    transitions = {unit for r in valid for unit in verified_search_units(r, 'transition')}
    applicable = [r for r in valid if r.get('relation') and r.get('mr_verdict', r.get('verdict')) in ('pass', 'suspected_failure')]
    pairs = {(r['api'], r['relation']) for r in applicable}
    state_relations = {(unit, r['relation']) for r in applicable
                       for unit in verified_search_units(r, include_optimizer=False)}
    elapsed = report.get('elapsed_seconds', 0)
    usage = [r.get(key) or {} for r in rows for key in ('planning_usage', 'generation_usage')
             if (key in r or key == 'generation_usage' and r.get('first_blocking_stage') != 'state_plan')
             and r.get('planning_dispatched' if key == 'planning_usage' else 'program_generation_dispatched') is not False]
    planning_rows = [r for r in rows if r.get('planning_dispatched', 'planning_usage' in r) is True]
    known_tokens = sum(u.get('input_tokens') or 0 for u in usage) + sum(u.get('output_tokens') or 0 for u in usage)
    tokens_complete = bool(rows) and all(u.get('input_tokens') is not None and u.get('output_tokens') is not None for u in usage)
    audits = [a for r in rows for a in r.get('gradient_audits', {}).values()]
    return {'attempts':len(rows), 'runtime_valid':len(valid),
            'pipeline_timing': report.get('pipeline_timing'),
            'pipeline_complete': report.get('pipeline_complete'),
            'optimizer_chain': summarize_optimizer_search(rows, elapsed),
            'exploration_oracle': summarize_exploration_oracle(report, elapsed, known_tokens, tokens_complete),
            'numerical_audit_statuses':dict(Counter(a.get('status', 'unknown') for a in audits)),
            'numerical_audit_seconds':sum(a.get('cost', {}).get('seconds', 0) for a in audits),
            'numerical_unknown_reasons':dict(Counter(a.get('reason', 'unspecified') for a in audits if a.get('status') in ('unknown','unsupported'))),
            'numerically_reported_cases':sum(bool(r.get('numerical_violation_members')) for r in rows),
            'planning_calls':len(planning_rows),
            'planning_family_calls':dict(Counter(r['planning_family_selection']['family'] for r in planning_rows
                                                if 'planning_family_selection' in r)),
            'planning_unresolved_proposals':sum(len(r.get('planning_unresolved', [])) for r in rows),
            'planning_outcomes':dict(Counter(r['planning_outcome'] for r in rows if r.get('planning_outcome'))),
            'repair_stop_reasons':dict(Counter(r['repair_stopped'] for r in rows if r.get('repair_stopped'))),
            'program_generation_calls':sum(r.get('program_generation_dispatched', 'generation_usage' in r) for r in rows),
            'configuration_reuse_cases':sum(r.get('source_reused') is True for r in rows),
            'proposal_origins':dict(Counter(r.get('proposal_origin', 'fresh_or_legacy') for r in rows)),
            'library_planning':report.get('library_planning', {}),
            'library_configuration_cases':sum(r.get('proposal_origin') == 'library_known_action' for r in rows),
            'library_configuration_outcomes':dict(Counter(r['library_configuration_outcome'] for r in rows
                                                        if 'library_configuration_outcome' in r)),
            'planning_rejections':sum(r.get('first_blocking_stage') == 'state_plan' for r in rows),
            'planned_executions':sum('state_plan' in r or bool(r.get('configuration', {}).get('predicates')) for r in rows),
            'goal_attained_cases':sum(bool(r.get('inspection', {}).get('predicates')) and r['inspection'].get('target_state_reached', False) for r in valid),
            'goal_attained_applicable_cases':sum(bool(r['inspection'].get('predicates')) and r['inspection'].get('target_state_reached', False) for r in applicable),
            'valid_cases_per_second':len(valid)/elapsed if elapsed else None,
            'runtime_valid_rate':len(valid)/len(rows) if rows else None,
            'verified_apis':sorted({r['api'] for r in valid}),
            'semantic_units':len(semantic), 'transition_units':len(transitions),
            'semantic_coverage_version': SEMANTIC_COVERAGE_VERSION,
            'observed_coverage_versions': dict(Counter(
                str(r.get('inspection', {}).get('semantic_coverage', {}).get('version', 'missing')) for r in rows)),
            'unary_semantic_units': unit_kinds['unary'], 'joint_semantic_units': unit_kinds['joint'],
            'joint_measurement_counts': dict(measurement_counts),
            'relational_goal_attempts': len(relational_goals),
            'relational_goal_attained_cases': relational_attained,
            'relational_goal_attainment_rate': relational_attained / len(relational_goals) if relational_goals else None,
            'relational_goal_denominator': 'attempts requesting a tensor relation, including generation and execution failures',
            'joint_units_per_second': unit_kinds['joint']/elapsed if elapsed else None,
            'joint_units_per_1000_tokens': 1000*unit_kinds['joint']/known_tokens if tokens_complete and known_tokens else None,
            'joint_coverage_denominator': None,
            'joint_measurement_scope': 'bounded per-execution observations; counts are not distinct feasible candidates or theoretical coverage',
            'applicable_comparisons':len(applicable), 'api_mr_units':len(pairs),
            'applicable_comparisons_per_second':len(applicable)/elapsed if elapsed else None,
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
            'candidate_denominator':'per-decision known candidates after history and budget filters; feasibility unproven, no theoretical coverage fraction',
            'confirmed_defects':None, 'stop_reason':report.get('stop_reason')}


def summarize_multi_api(sessions, scheduling, limits, wall_seconds):
    """Keep per-session oracle/coverage denominators; aggregate orchestration only."""
    from copy import deepcopy
    sessions = deepcopy(sessions)
    reports = [row['report'] for row in sessions if row.get('report') is not None]
    complete = all(row['state'] == 'completed' for row in sessions)
    pending = any(row['state'] in ('queued', 'running') for row in sessions)
    reason = ('interrupted' if scheduling['cancelled'] else 'running' if pending else
              'completed' if complete else 'completed_with_failures')
    strategies = {row['strategy'] for row in sessions}
    return dict(strategy=next(iter(strategies)) if len(strategies) == 1 else 'multi_api',
                multi_api=True, stop_reason=reason, sessions=sessions, per_api=reports,
                expected_apis=[row['api'] for row in sessions],
                completed_api_runs=sum(row['state'] == 'completed' for row in sessions),
                elapsed_seconds=wall_seconds,
                per_api_elapsed_seconds_sum=sum(row.get('session_wall_seconds', 0.) for row in sessions),
                scheduling=scheduling, resource_limits=limits,
                coverage_scope='Per-session verified denominators; no synthesized global coverage or MR Pass',
                orchestration_scope='Completed sessions describe controller completion, not framework correctness')

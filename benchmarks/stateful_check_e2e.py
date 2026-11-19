"""Frozen ten-API fusion campaign with independent per-API model budgets."""
import argparse
from collections import Counter
from dataclasses import asdict
from itertools import combinations
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

from e2e_latency import digest, read, targets, write


PROJECT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen3.8-27B"
REVISION = "37561816e66b5b9153b1b64c10968c18c1e6f4a0"


def hashes(directory):
    return {str(p.relative_to(directory)): digest(p) for p in sorted(directory.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    snapshot = root / 'snapshot'
    for subdir in ('src', 'benchmarks', 'tests', 'sbatch/lib'):
        shutil.copytree(PROJECT / subdir, snapshot / subdir,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    launcher = 'sbatch/experiments/stateful_check_e2e.sbatch'
    (snapshot / launcher).parent.mkdir()
    shutil.copy2(PROJECT / launcher, snapshot / launcher)
    validation = 'sbatch/validation/evidence_guided.sbatch'
    (snapshot / validation).parent.mkdir()
    shutil.copy2(PROJECT / validation, snapshot / validation)
    protocol_validation = 'sbatch/validation/oracle_precondition.sbatch'
    shutil.copy2(PROJECT / protocol_validation, snapshot / protocol_validation)
    roster = targets(read(snapshot / 'benchmarks/pytorch_correctness/candidates.json'))
    protocol = dict(version='stateful-check-e2e-1', targets=roster, seed=0, model_seconds_per_api=180,
                    workers=8, generation_concurrency=2, active_apis=len(roster), cases=100000,
                    worker_timeout=120, model=MODEL, revision=REVISION, temperature=0., top_p=1.,
                    max_tokens=4096, max_model_len=16384, mtp_tokens=3,
                    allocation=dict(partition='8gpus', nodes=1, cpus=16, gpus=2, memory_gib=192,
                                    wall_seconds=43200, final_audit_and_cleanup_reserve_seconds=1800,
                                    rationale='One service GPU and four shared resident CUDA workers for two-device '
                                              'drain after the independent per-API generation budgets'),
                    differential_workers=dict(cuda=4, cpu=2, comparison=1),
                    scheduler='independent API sessions, round robin with unused capacity borrowing',
                    e2e_scope='API admission to check exit, including spec, preflight, library overhead, shared '
                              'scheduler waiting, generation, representation drain, online CPU/CUDA workers, '
                              'their final join and reconciliation. Parallel durations are not added to wall time. '
                              'Campaign wall excludes offline attribution and vLLM startup.',
                    snapshot_sha256=hashes(snapshot))
    write(root / 'protocol.json', protocol)
    (root / 'protocol.sha256').write_text(digest(root / 'protocol.json') + '\n')
    print(root / 'protocol.json', flush=True)


def verify(root):
    protocol = read(root / 'protocol.json')
    if digest(root / 'protocol.json') != (root / 'protocol.sha256').read_text().strip():
        raise ValueError('Frozen campaign protocol changed')
    if hashes(root / 'snapshot') != protocol['snapshot_sha256']:
        raise ValueError('Frozen campaign inputs changed')
    return protocol


def distribution(values):
    ordered = sorted(values)
    return dict(count=len(ordered), mean=sum(ordered)/len(ordered) if ordered else None,
                p50=ordered[math.ceil(len(ordered)*.5)-1] if ordered else None,
                p95=ordered[math.ceil(len(ordered)*.95)-1] if ordered else None,
                method='nearest-rank percentile')


def differential_timing(rows):
    submitted, finished, latencies = [], [], []
    rounds = required = missing = dual = 0
    for row in rows:
        if not row.get('legal_member'):
            continue
        required += 1
        timing = row.get('timing') or {}
        if 'submitted' not in timing or 'updated' not in timing:
            missing += 1
            continue
        submitted.append(timing['submitted'])
        finished.append(timing['updated'])
        rounds += len(timing['rounds'])
        dual += all('0-' + d in row.get('devices', {}) for d in ('cpu', 'cuda'))
        if row['status'] != 'incomplete':
            latencies.append(timing['updated'] - timing['submitted'])
    span = max(finished)-min(submitted) if submitted else None
    return dict(required_members=required, members_with_both_side_records=dual,
                missing_timing=missing, completed_comparisons_including_replays=rounds,
                elapsed_first_admission_to_last_update_seconds=span,
                compared_pairs_per_second=rounds/span if span and span > 0 else None,
                terminal_member_latency_seconds=distribution(latencies))


def request_times(directory):
    # Pipeline events include HTTP failures, timeouts and the timing basis.
    ledger = directory / 'execution/pipeline-state.json'
    if not ledger.is_file():
        return {}
    state = read(ledger)
    completed = [e for e in state.get('model_events', []) if e.get('finished') is not None]
    times = [e['seconds'] for e in completed]
    latency = distribution(times)
    truncated = [e for e in completed if e['outcome'] == 'generation_truncated']
    retries = [e for e in completed if e.get('retry_of_attempt') is not None]
    gap_requests = [e for e in completed if e.get('oracle_precondition')]
    return dict(requests=len(times), request_seconds_sum=sum(times),
                truncated_requests=len(truncated), retry_requests=len(retries),
                truncated_seconds=sum(e['seconds'] for e in truncated),
                retry_seconds=sum(e['seconds'] for e in retries),
                oracle_precondition_requests=len(gap_requests),
                oracle_precondition_request_seconds=sum(e['seconds'] for e in gap_requests),
                truncated_completion_tokens=[(e.get('usage') or {}).get('completion_tokens') for e in truncated],
                latency_mean=latency['mean'], latency_p50=latency['p50'], latency_p95=latency['p95'],
                outcomes=dict(Counter(e['outcome'] for e in completed)),
                budget_seconds=state['budget'], charged_seconds=state['model_seconds'],
                budget_stop=state.get('generation_budget_stop'))


def target_returned(directory):
    trace = directory / 'execution/source-trace.jsonl'
    if not trace.is_file():
        return False
    with trace.open() as lines:
        for line in lines:
            # Active traces can end with a partial append.
            if line.endswith('\n') and json.loads(line).get('event') == 'api_exit':
                return True
    return False


def published(path):
    """Startup probes may race direct JSON writes in an active provider process."""
    try:
        return read(path)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def started(job):
    service = job / 'service-ready.json'
    environment = job / 'serving/environment.json'
    if not service.is_file() or not environment.is_file():
        return dict(started=False, reason='service_not_ready')
    ready, served = published(service), published(environment)
    if (ready.get('model_list_verified') is not True or ready.get('health_http_status') != 200
            or served.get('model') != MODEL or served.get('declared_revision') != REVISION):
        return dict(started=False, reason='serving_identity_or_health_unverified')
    sessions = []
    for stamp in sorted((job / 'campaign').glob('check-start-*.json')):
        admission = read(stamp)
        api_dir = Path(admission['directory'])
        session_path = api_dir/'session.json'
        session_end = read(session_path).get('finished_epoch') if session_path.is_file() else None
        execution = api_dir / 'execution'
        verified_request = None
        for process in execution.glob('provider-*/process.json'):
            if (published(process).get('outcome') != 'completed'
                    or published(process.parent/'response.json').get('model') != MODEL):
                continue
            for artifact in process.parent.glob('http-*.json'):
                http = published(artifact)
                if http.get('http_status') == 200 and http['request']['model'] == MODEL:
                    verified_request = str(artifact)
                    break
            if verified_request:
                break
        if verified_request:
            match = next((p for p in sorted(execution.glob('attempt-*')) if target_returned(p)), None)
            if match:
                sessions.append(dict(api=admission['api'], started_epoch=admission['started_epoch'],
                    finished_epoch=session_end, request=verified_request, target_trace=str(match/'execution/source-trace.jsonl')))
    now = time.time()
    overlap = any(max(a['started_epoch'], b['started_epoch']) < min(a['finished_epoch'] or now,b['finished_epoch'] or now)
                  for a,b in combinations(sessions,2))
    return dict(started=overlap, overlapping_target_sessions=overlap, serving=ready, target_returned=sessions)


def case_paths(execution):
    yield from sorted(execution.glob('attempt-*/outcome.json'))
    yield from sorted(execution.glob('representations/*/*/run-*/outcome.json'))


def differential_cost(directory):
    """A resumed task can retain multiple trials. Each isolated step owns one process.json."""
    root = directory / 'differential'
    trials = sorted({*root.glob('*/trial-*'), *root.glob('*/*/trial-*')})
    seconds = 0.
    workers = missing = 0
    for trial in trials:
        for request in trial.glob('*/request.json'):
            process = request.parent/'process.json'
            if process.is_file():
                seconds += read(process)['elapsed_seconds']
                workers += 1
            else:
                missing += 1
    return dict(trials=len(trials), recorded_processes=workers, unmeasured_processes=missing,
                worker_seconds=seconds, complete=not missing)


def score_session(session, target, audit_deadline):
    from mirage.artifacts.bundle import load_tensor_record
    from mirage.artifacts.report import verified_search_units
    from mirage.generator.state_plan import state_family_catalog
    from pytorch_correctness.evaluate import stateful_attribution, stateful_independent_audit
    directory = Path(session['directory'])
    execution = directory / 'execution'
    totals = Counter()
    mechanisms = Counter()
    gaps_seen = set()
    route_funnel = Counter(dict(planned=0, dispatched=0, legal=0, use_witnessed=0,
                                applicable=0, violations=0, independently_replayed=0))
    spec = directory/'api-spec.json'
    spec_data = read(spec) if spec.exists() else {}
    resolved = spec_data.get('status') == 'resolved'
    catalog = state_family_catalog(spec_data, target['device']) if resolved else []
    families = {goal['id']:goal['family'] for goal in catalog}
    observed_states, joint_states, transitions = set(), set(), set()
    capabilities = set()
    inspected_transitions, inspected_combinations, selected_goals = set(), set(), set()
    matches = []
    errors = []
    reviewed = directory / 'benchmark-audit.jsonl'
    def audit(path, attempt, reproduced, *, verify_only=False):
        totals['expected_artifacts'] += 1
        if not path.is_file():
            errors.append(dict(attempt=attempt, file=str(path), error='expected_tensor_artifact_missing'))
            return False
        if verify_only:
            totals['existence_only_members'] += 1
            stream.write(json.dumps(dict(attempt=attempt, file=str(path), disposition='existence_only')) + '\n')
            return False
        if time.monotonic() >= audit_deadline:
            totals['unaudited_deadline_members'] += 1
            stream.write(json.dumps(dict(attempt=attempt, file=str(path), disposition='audit_deadline')) + '\n')
            return False
        try:
            tensor_record = load_tensor_record(path)
            independent = stateful_independent_audit(target['id'], tensor_record)
            result = stateful_attribution(target['id'], tensor_record, independent)
            mechanism = result['structural_trigger']
            mechanisms['unmeasured' if mechanism is None else 'observed' if mechanism else 'absent'] += 1
            credited = bool(reproduced and result['benchmark_reproduction'])
            detail = dict(attempt=attempt, file=str(path), independent=independent, attribution=result,
                          independently_replayed=reproduced, credited=credited)
            stream.write(json.dumps(detail, allow_nan=False) + '\n')
            if credited:
                matches.append(dict(attempt=attempt, file=str(path)))
            totals['audited_members'] += 1
            return result['benchmark_reproduction']
        except (IndexError, KeyError, TypeError, ValueError, RuntimeError, OSError, OverflowError) as exc:
            errors.append(dict(file=str(path), error=str(exc)))
            return False
    directory.mkdir(parents=True, exist_ok=True)
    with reviewed.open('x') as stream:
        for outcome in case_paths(execution):
            case = read(outcome)
            attempt = case['attempt']
            returned = target_returned(outcome.parent)
            generated = case.get('proposal_origin') != 'framework_representation'
            totals['target_return_cases'] += returned
            totals['target_return_generated_cases'] += bool(generated and returned)
            totals['source_attempts'] += generated
            totals['legal_sources'] += bool(generated and case.get('source_completed') and case.get('api_executed'))
            totals['representation_attempts'] += not generated
            transitions.update(verified_search_units(case, 'transition'))
            goal = case.get('configuration', {}).get('state_goal')
            if generated and goal:
                selected_goals.add(goal['id'])
                from mirage.generator.state_plan import observed_capabilities
                capabilities.update(observed_capabilities(target['api'], [case]))
                if (case.get('inspection', {}).get('state_goal') or {}).get('attained'):
                    totals['selected_state_attained_sources'] += 1
                if case.get('selection', {}).get('confirm_prefix') and case.get('inspection', {}).get('valid'):
                    totals['confirmed_prefix_sources'] += 1
            for name in ('inspection','followup_inspection'):
                inspection = case.get(name) or {}
                if not inspection.get('legal_source'):
                    continue
                inspected_transitions.update(inspection.get('training_state_transitions', ()))
                inspected_combinations.update(tuple(pair) for pair in inspection.get('training_state_combinations', ()))
                scoped = {}
                for unit, fact in inspection.get('training_state_evidence', {}).items():
                    observed_states.add(unit)
                    scoped.setdefault((fact['call'],fact['checkpoint']), []).append(unit)
                for units in scoped.values():
                    joint_states.update((a,b) for a,b in combinations(sorted(units),2)
                                        if a in families and b in families and families[a] != families[b])
            replay_file = outcome.parent / 'automatic-replays.json'
            replays = read(replay_file) if replay_file.exists() else case.get('automatic_replays', [])
            suspect = case.get('verdict') == 'suspected_failure'
            stable = suspect and any(r.get('replay_valid') is True for r in replays)
            inspection = case.get('inspection') or {}
            for gap in inspection.get('product_use', {}).get('oracle_gaps', ()):
                gaps_seen.add((gap['oracle'], gap['required_use']))
            if generated and goal and goal.get('required_use'):
                route_funnel['planned'] += 1
                route_funnel['dispatched'] += case.get('program_generation_dispatched') is True
                route_funnel['legal'] += bool(inspection.get('legal_source'))
                witnessed = any(u['mode'] == goal['required_use']
                                for u in inspection.get('product_use', {}).get('uses', ()))
                route_funnel['use_witnessed'] += witnessed
                applicable = (witnessed and case.get('relation') == goal['oracle']
                              and case.get('mr_verdict') in ('pass','suspected_failure'))
                violated = applicable and case.get('mr_verdict') == 'suspected_failure'
                route_funnel['applicable'] += applicable
                route_funnel['violations'] += violated
                route_funnel['independently_replayed'] += bool(violated and stable)
            totals['primary_suspects'] += suspect
            totals['primary_replayed_suspects'] += stable
            for member, key in (('source','inspection'), ('followup','followup_inspection')):
                inspection = case.get(key) or {}
                path = outcome.parent / 'execution' / (member + '.pt')
                if (case.get(member + '_completed') or inspection.get('legal_source') or inspection.get('valid')
                        or path.is_file()):
                    legal = inspection.get('legal_source') or inspection.get('valid')
                    if member == 'source' and case.get('source_completed') and case.get('api_executed'):
                        legal = True
                    if legal and not inspection.get('valid'):
                        totals['legal_state_unmet_members'] += 1
                    # Credit a primary replay only under its actual member's valid claim.
                    audit(path, attempt, bool(stable and inspection.get('valid')), verify_only=not legal)
        diff_path = directory / 'differential-ledger.json'
        differential = read(diff_path) if diff_path.exists() else []
        for result in differential:
            if result['status'] == 'stable_mismatch':
                for device in ('cpu', 'cuda'):
                    trial = Path(result['directory'])
                    corroborated = audit(trial / ('1-' + device) / 'member.pt', result['attempt'], False)
                    audit(trial / ('0-' + device) / 'member.pt', result['attempt'], corroborated)
    counts = Counter(r['status'] for r in differential)
    reasons = Counter(r.get('reason') or 'unknown' for r in differential
                      if r['status'] not in ('agreement', 'stable_mismatch'))
    observed = Counter()
    for row in differential:
        comparison = row.get('comparison') or {}
        raw = comparison.get('raw', {})
        observed['members_with_fields'] += bool(raw.get('fields'))
        observed['members_with_differences'] += bool(raw.get('differences'))
        observed['stable_candidates'] += bool(comparison.get('stable_observed_fields'))
        observed['oracle_mismatches_replayed'] += row.get('status') == 'stable_mismatch'
    sides = Counter()
    for row in differential:
        for device in ('cpu', 'cuda'):
            side = row.get('devices', {}).get('0-' + device, {})
            if side.get('execution_attempted') is True:
                sides[device + '_attempted'] += 1
            if side.get('completed') is True:
                sides[device + '_completed'] += 1
            if side.get('target_returned') is True:
                sides[device + '_target_returned'] += 1
    traversal_file = execution / 'representation-traversal.json'
    traversed = read(traversal_file).get('sources', []) if traversal_file.exists() else []
    repr_counts = Counter(c['status'] for s in traversed for c in s['candidates'])
    eligible = sum(c.get('eligibility', {}).get('status') == 'verified'
                   for s in traversed for c in s['candidates'])
    summary_file = directory / 'summary.json'
    summary = read(summary_file) if summary_file.is_file() else {}
    process_cost = differential_cost(directory)
    if resolved:
        catalog = state_family_catalog(spec_data, target['device'], observed=capabilities)
    proposed = {g['id'] for g in catalog if g['availability'] == 'candidate'}
    compound = {g['id'] for g in catalog if g['id'] in proposed and len(g['prerequisites']) >= 2}
    def denominator(eligible):
        if not resolved:
            return None
        return dict(proposed=len(eligible), selected=len(eligible & selected_goals),
                    witnessed=len(eligible & observed_states), unwitnessed=sorted(eligible-observed_states))
    return dict(api=target['api'], issue=target['id'], session=session['session_id'], state=session['state'],
        session_reason=session.get('reason'), check_status=summary.get('status'), check_reason=summary.get('reason'),
        e2e_seconds=session.get('session_wall_seconds'), queue_seconds=session.get('admission_queue_seconds'),
        submission_to_end_seconds=session.get('session_wall_seconds',0)+session.get('admission_queue_seconds',0)
                                  if 'session_wall_seconds' in session else None,
        generation_completion=summary.get('generation_completion'), generation=request_times(directory), differential_timing=differential_timing(differential), metrics=summary.get('metrics') or {},
        representation=dict(inventory_available=traversal_file.exists(),
                            registered_sources=len(traversed) if traversal_file.exists() else None,
                            eligible=eligible if traversal_file.exists() else None, statuses=dict(repr_counts)),
        spec_status=spec_data.get('status','unmeasured'), state_catalog=catalog if resolved else None,
        state_goal_coverage=denominator(proposed), multi_prerequisite_goal_coverage=denominator(compound),
        coverage_scope='Finite goals admitted by the frozen planner, including unselected and unachieved '
                       'goals; runtime feasibility of those goals is unknown. Incidental observer transitions '
                       'and combinations are reported as distinct witnessed units with no theoretical denominator.',
        observed_states=sorted(observed_states), oracle_precondition_funnel=dict(route_funnel,
            distinct_candidate_routes=len(gaps_seen)),
        observed_families=sorted({families[u] for u in observed_states if u in families}),
        training_state_transitions=sorted(inspected_transitions), training_state_combinations=sorted(inspected_combinations),
        cross_family_same_call_and_checkpoint=sorted(joint_states), distinct_generic_transition_signatures=len(transitions),
        case_counts=dict(totals), differential_observations=dict(observed), differential_counts=dict(counts), differential_sides=dict(sides),
        differential_reasons=dict(reasons),
        differential_required=sum(r.get('legal_member') is True for r in differential),
        differential_paired_target_returns=sum(all(r.get('devices', {}).get('0-' + d, {}).get('target_returned')
                                                   for d in ('cpu','cuda')) for r in differential),
        differential_worker_seconds=process_cost['worker_seconds'], differential_all_trials=process_cost,
        differential_latest_trials_worker_seconds=sum(p['elapsed_seconds'] for r in differential for p in r.get('processes', [])),
        differential_finalize_seconds=summary.get('differential', {}).get('elapsed_seconds'),
        mechanisms=dict(mechanisms), benchmark_reproduced=bool(matches), benchmark_matches=matches,
        attribution_complete=not errors and not totals['unaudited_deadline_members'] and summary.get('terminal') is True
                             and summary.get('status') in ('completed','completed_with_generation_errors')
                             and totals['expected_artifacts'] == totals['audited_members'] + totals['existence_only_members'],
        attribution_scope='All completed legal source and followup members, including unmet state claims. '
                          'Only independently positive, valid primary replays or a repeated positive on the '
                          'same side of a stable CPU/CUDA mismatch receive attribution. Failed, skipped or '
                          'incomplete sessions retain available partial evidence.',
        evidence_errors=errors, evidence=str(reviewed), report=str(summary_file))


def campaign(root, job, seconds, overall_seconds):
    from mirage.contracts.programs import RunOptions
    from mirage.contracts.scheduling import APIJob, SchedulingOptions
    from mirage.runtime.multi_api import run_sessions
    if not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Campaign requires a Slurm step')
    protocol = verify(root)
    if PROJECT != (root/'snapshot').resolve():
        raise ValueError('Start the frozen campaign driver')
    serving = read(job/'serving/environment.json')
    if (serving.get('model'), serving.get('declared_revision'), serving.get('mtp_tokens')) != (
            protocol['model'], protocol['revision'], protocol['mtp_tokens']):
        raise ValueError('Measured service differs from frozen protocol')
    if seconds <= 0 or overall_seconds <= seconds:
        raise ValueError('No time remaining for the campaign')
    audit_deadline = time.monotonic() + overall_seconds - 30
    endpoint = job / 'campaign'
    jobs = []
    for i, target in enumerate(protocol['targets']):
        identifier = f'api-{i:03d}'
        options = RunOptions(output=str(endpoint/identifier), provider='vllm', strategy='stateful',
            contract_version=4, device=target['device'], time_limit=protocol['model_seconds_per_api'],
            cases=protocol['cases'], seed=protocol['seed'], workers=protocol['workers'],
            generation_concurrency=protocol['generation_concurrency'], worker_timeout=protocol['worker_timeout'],
            generation_pipeline=True, generation_overlap=True, representation_traversal=True, oracle_improvements=True)
        jobs.append(APIJob(identifier, target['api'], options, check=True))
    limits = SchedulingOptions(protocol['workers'], protocol['generation_concurrency'], protocol['active_apis'])
    write(job / 'campaign-start.json', dict(started_epoch=time.time(), job=os.environ['SLURM_JOB_ID'],
          step=os.environ['SLURM_STEP_ID'], protocol=str(root/'protocol.json'), limits=asdict(limits)))
    campaign_report = run_sessions(jobs, limits, endpoint, deadline=time.monotonic()+seconds)
    audit_started = time.monotonic()
    measured = []
    for row, target in zip(campaign_report['sessions'], protocol['targets'], strict=True):
        metrics = score_session(row, target, audit_deadline)
        write(Path(row['directory'])/'benchmark-metrics.json', metrics)
        measured.append(metrics)
        write(job/'benchmark-progress.json', dict(audited=len(measured), expected=len(jobs)))
    def selected(fn):
        return [r['api'] for r in measured if fn(r)]
    scheduling = campaign_report['scheduling']
    aggregate = dict(protocol_sha256=digest(root/'protocol.json'), roster=protocol['targets'],
        expected_apis=len(jobs), measured_sessions=len(measured), audit_seconds=time.monotonic()-audit_started,
        campaign_e2e_seconds=campaign_report['elapsed_seconds'], stop_reason=campaign_report['stop_reason'],
        differential_timing=differential_timing(row for session in campaign_report['sessions']
            for ledger in [Path(session['directory'])/'differential-ledger.json'] if ledger.exists()
            for row in read(ledger)),
        scheduling_peaks=scheduling['peaks'], per_session_resources=scheduling['session_resource_totals'],
        truncated_requests=sum(r['generation'].get('truncated_requests',0) for r in measured),
        retry_requests=sum(r['generation'].get('retry_requests',0) for r in measured),
        retry_seconds=sum(r['generation'].get('retry_seconds',0) for r in measured),
        generation_seconds_sum=sum(r['generation'].get('charged_seconds',0) for r in measured),
        per_api_e2e_seconds=distribution(r['e2e_seconds'] for r in measured if r['e2e_seconds'] is not None),
        target_returned_apis=selected(lambda r: r['case_counts'].get('target_return_cases',0)>0),
        runtime_valid_apis=selected(lambda r: r['metrics'].get('runtime_valid',0)>0),
        state_goals_by_api=dict(resolved_specs=sum(r['state_goal_coverage'] is not None for r in measured),
            unknown_spec_apis=selected(lambda r: r['state_goal_coverage'] is None),
            proposed=sum((r['state_goal_coverage'] or {}).get('proposed',0) for r in measured),
            witnessed=sum((r['state_goal_coverage'] or {}).get('witnessed',0) for r in measured)),
        state_composite_goals_by_api=dict(
            proposed=sum((r['multi_prerequisite_goal_coverage'] or {}).get('proposed',0) for r in measured),
            witnessed=sum((r['multi_prerequisite_goal_coverage'] or {}).get('witnessed',0) for r in measured)),
        api_training_state_transition_units=sum(len(r['training_state_transitions']) for r in measured),
        api_training_state_combination_units=sum(len(r['training_state_combinations']) for r in measured),
        benchmark_reproduced_apis=selected(lambda r:r['benchmark_reproduced']),
        attribution_incomplete_apis=selected(lambda r: not r['attribution_complete']),
        differential_observations=dict(sum((Counter(r['differential_observations']) for r in measured), Counter())),
        differential_counts=dict(sum((Counter(r['differential_counts']) for r in measured), Counter())),
        sessions=measured, service=read(job / 'service-ready.json'))
    write(job/'benchmark-results.json', aggregate)
    return int(campaign_report['stop_reason'] != 'completed' or any(not r['attribution_complete'] for r in measured))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for cmd in ('prepare','verify','started'):
        commands.add_parser(cmd).add_argument('root', type=Path)
    run = commands.add_parser('campaign')
    run.add_argument('root', type=Path)
    run.add_argument('job', type=Path)
    run.add_argument('seconds', type=int)
    run.add_argument('overall_seconds', type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == 'prepare':
        prepare(root)
    elif args.command == 'verify':
        verify(root)
    elif args.command == 'started':
        evidence = started(root)
        print(json.dumps(evidence))
        return 0 if evidence['started'] else 2
    else:
        return campaign(root, args.job.resolve(), args.seconds, args.overall_seconds)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


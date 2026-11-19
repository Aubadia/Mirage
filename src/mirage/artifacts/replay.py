"""Run frozen evidence through isolated workers without model requests."""
from dataclasses import asdict
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time
import uuid

from mirage.artifacts.bundle import digest, verify
from mirage.runtime.process import run_worker
from mirage.runtime import differential_pool
from mirage.workflow import replay, write_json

__all__ = ["replay"]


@lru_cache(maxsize=32)
def frozen_engine_inventory(engine, manifest_version):
    if manifest_version:
        files = json.loads((engine.parent/'engine-sha256.json').read_text())
        return {p:h for p,h in files.items() if p.endswith('.py')}
    return {str(f.relative_to(engine)): digest(f) for f in sorted(engine.rglob('*.py'))}


@lru_cache(maxsize=8192)
def _file_digest(path, version):
    return digest(path)


def _version(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_ctime_ns, stat.st_size


def primary_route(row):
    if (not row.get('inspection', {}).get('valid') or row.get('mutation_obligations_valid') is False
            or (row.get('proposal_origin') == 'framework_representation'
                and not row.get('representation_candidate_attained'))):
        return 'invalid'
    if row.get('mr_verdict') in ('pass', 'suspected_failure'):
        return 'primary_mr'
    if row.get('numerical_violation_members'):
        return 'primary_numerical'
    if row.get('configuration', {}).get('check_claim') in ('backward', 'accumulation') and any(
            audit.get('status') in ('agreement', 'violation') for audit in row.get('gradient_audits', {}).values()):
        return 'primary_numerical'
    return 'fallback'


def freeze_differential(root, row, schemas, *, member, engine, engine_inventory, legacy=False):
    from mirage.inspector.comparison import DEVICE_POLICY, PAIRED_DEVICE_POLICY, OBSERVED_DEVICE_POLICY
    directory = Path(row['directory'])
    source, original = directory/'program.py', directory/'execution'/f'{member}.pt'
    if not original.is_file():
        return None
    attempt = row['attempt']
    key = (f'attempt-{attempt:04d}' if isinstance(attempt, int) else
           'case-' + hashlib.sha256(str(attempt).encode()).hexdigest())
    dest = root/'differential'/key
    if not legacy:
        dest = dest/member
    dest.mkdir(parents=True, exist_ok=True)
    path = dest/'manifest.json'
    observed = not legacy and 'mirage/inspector/device_observations.py' in engine_inventory
    manifest = dict(format='device-differential-1' if legacy else 'device-differential-3' if observed else 'device-differential-2',
                    policy=DEVICE_POLICY if legacy else OBSERVED_DEVICE_POLICY if observed else PAIRED_DEVICE_POLICY,
                    engine=str(engine), engine_inventory=engine_inventory,
                    source=str(source), original=str(original),
                    sha256={str(f): digest(f) for f in (source, original)}, schemas=schemas,
                    primary_verdict=row.get('mr_verdict'), primary_reason=row.get('reason'))
    if not legacy:
        manifest.update(member=member, attempt=attempt,
                        source_device=row.get('configuration', {}).get('device'),
                        api=row.get('api', row.get('configuration', {}).get('api')))
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError('Differential manifest changed')
        return path
    write_json(path, manifest)
    return path


def replay_devices(path, *, timeout=60, deadline=None, live=False, admitted=None):
    path = Path(path)
    manifest = json.loads(path.read_text())
    from mirage.inspector.comparison import DEVICE_POLICY, PAIRED_DEVICE_POLICY, OBSERVED_DEVICE_POLICY
    policies = {'device-differential-1': DEVICE_POLICY, 'device-differential-2': PAIRED_DEVICE_POLICY,
                'device-differential-3': OBSERVED_DEVICE_POLICY}
    if manifest.get('format') not in policies or manifest['policy'] != policies[manifest['format']]:
        raise ValueError('Unknown device comparison policy')
    legacy = manifest['format'] == 'device-differential-1'
    engine = Path(manifest['engine'])
    if live:
        for name, checksum in manifest['engine_inventory'].items():
            source = (engine/name).resolve()
            if not source.is_relative_to(engine.resolve()) or _file_digest(source, _version(source)) != checksum:
                raise ValueError(f'Differential engine changed: {name}')
    else:
        verify(engine, manifest['engine_inventory'])
    for filename, expected in manifest['sha256'].items():
        if digest(Path(filename)) != expected:
            raise ValueError('Differential evidence changed')
    root = path.parent/('trial-' + uuid.uuid4().hex)
    root.mkdir()
    report = dict(status='incomplete', directory=str(root), manifest=str(path), processes=[],
                  primary_verdict=manifest['primary_verdict'], primary_reason=manifest['primary_reason'],
                  provider_calls=0, confirmed_defect=False)
    if not legacy:
        report.update(member=manifest['member'], policy=manifest['policy'], devices={})
    pooled = manifest['format'] == 'device-differential-3' and differential_pool.scope() is not None
    if pooled:
        identity = hashlib.sha256(json.dumps(manifest['engine_inventory'], sort_keys=True).encode()).hexdigest()
        report['timing'] = dict(submitted=admitted if admitted is not None else time.monotonic(), rounds=[])
    fresh_after = 0.
    def save(status, reason=None):
        report.update(status=status, reason=reason)
        if pooled:
            report['timing']['updated'] = time.monotonic()
        write_json(root/'summary.json', report)
        return report
    save('incomplete', 'trial_started')
    def dispatch(label, request):
        if pooled:
            request.update(engine_identity=identity, fresh_after=fresh_after if request['mode']=='device_member' else 0)
        remaining = timeout if deadline is None else min(timeout, deadline-time.monotonic())
        process = run_worker(dict(request, engine=manifest['engine']), root/label, max(0, remaining))
        return process
    def collect(label, process):
        report['processes'].append(asdict(process))
        save('incomplete', label)
        if process.outcome != 'completed':
            save('incomplete', label + '_' + process.outcome)
            return None
        return json.loads((root/label/'summary.json').read_text())
    members = []
    for round_index in range(2):
        unavailable = []
        round_start = time.monotonic()
        def side(device):
            label = f'{round_index}-{device}'
            # A GPU-specific or mixed-device source can touch CUDA even during a CPU projection.
            resource = 'cuda' if manifest.get('source_device') == 'cuda' else device
            return dispatch(label, dict(mode='device_member', device=device, original=manifest['original'],
                source=Path(manifest['source']).read_text(), schemas=manifest['schemas'],
                config={'device': resource}, policy=manifest['policy']))
        if pooled:
            with ThreadPoolExecutor(max_workers=2) as sides:
                futures = [sides.submit(copy_context().run, side, d) for d in ('cpu','cuda')]
                processes = [f.result() for f in futures]
        else:
            processes = map(side, ('cpu','cuda'))
        for device, process in zip(('cpu','cuda'), processes):
            label = f'{round_index}-{device}'
            result = collect(label, process)
            if result is None or result['status'] != 'ready':
                if legacy:
                    return report if result is None else save(result['status'], result['reason'])
                if result is None:
                    result = dict(status='incomplete', reason=report['reason'], execution_attempted=None)
                unavailable.append(result)
            if not legacy:
                report['devices'][label] = result
                if result.get('alignment_reason') and manifest['policy'] != OBSERVED_DEVICE_POLICY:
                    unavailable.append(dict(status='incomparable', reason=result['alignment_reason']))
                save('incomplete', 'device_processed')
            if result['status'] != 'ready':
                continue
            members.append(str(root/label/'member.pt'))
        if unavailable:
            result = next((item for item in unavailable if item['status'] == 'incomplete'), unavailable[0])
            return save(result['status'], result.get('reason'))
        label = f'{round_index}-comparison'
        comparison = collect(label, dispatch(label, dict(mode='compare_devices', members=members, original=manifest['original'],
                            config={'device': 'cpu'}, policy=manifest['policy'], schemas=manifest['schemas'])))
        if comparison is None:
            return report
        report['comparison'] = comparison
        fresh_after = time.monotonic()
        if pooled:
            report['timing']['rounds'].append(dict(started=round_start, compared=fresh_after))
        if round_index or (comparison['status'] != 'mismatch' and not comparison.get('raw', {}).get('differences')):
            return save(comparison['status'], comparison.get('reason'))
    raise RuntimeError('Device comparison ended without a terminal result')


def differential_members(case, legacy=False):
    from mirage.sampler.representations import legal_check_source, previously_reached
    child = case.get('proposal_origin') == 'framework_representation'
    if legacy:
        return [('followup' if child else 'source')] if primary_route(case) == 'fallback' else []
    followup = case.get('followup_inspection') or {}
    legal_followup = case.get('followup_completed') is True and (
        followup.get('legal_source') is True or followup.get('valid') is True)
    if child:
        return ['followup'] if legal_followup or (case.get('representation_candidate_attained') is True and
                                case.get('mutation_obligations_valid') is True) else []
    if not legal_check_source(case) and not case.get('inspection', {}).get('valid'):
        return []
    members = ['source']
    plan = case.get('mutation') or {}
    if legal_followup or (all(k in plan for k in ('path', 'action', 'call_index')) and previously_reached(plan, case)):
        members.append('followup')
    return members


def assess_devices(root, result, spec, *, enabled, timeout, deadline=None, resume=False,
                   workers=1, engine=None, legacy=False, append=False, admitted=None):
    start = time.monotonic()
    root = Path(root)
    engine = Path(engine or root/'execution/engine').resolve()
    seal = engine.parent/'engine-sha256.json'
    inventory = frozen_engine_inventory(engine, _version(seal) if seal.is_file() else None)
    schemas = [item['schema'] for item in spec.get('operator_schemas', [])]
    rows, tasks = [], []
    for case in result.get('records', []):
        members = differential_members(case, legacy)
        for member in members or [None]:
            row = dict(attempt=case['attempt'], claim=case.get('configuration', {}).get('check_claim'), reason=None,
                       primary_route=primary_route(case), primary_reason=case.get('reason'),
                       status='not_requested')
            if not legacy:
                inspection = case.get('followup_inspection' if member == 'followup' else 'inspection', {})
                row.update(member=member, legal_member=member is not None,
                           requested_state_attained=inspection.get('valid') is True)
            rows.append(row)
            if member is None:
                continue
            if not enabled:
                row.update(status='disabled')
            elif legacy and not spec.get('environment', {}).get('cuda_available'):
                row.update(status='incomparable', reason='cuda_unavailable')
            elif legacy and row['claim'] not in ('forward', 'operand_relation'):
                row.update(status='incomparable', reason='claim_history_or_observables_unsupported')
            else:
                row.update(status='incomplete', reason='queued')
                tasks.append((len(rows)-1, case, member))
    ledger = root/'differential-ledger.json'
    def publish():
        if not append:
            write_json(ledger, rows)
            return
        from fcntl import flock, LOCK_EX
        with (root/'differential-ledger.lock').open('a') as lock:
            flock(lock, LOCK_EX)
            previous = json.loads(ledger.read_text()) if ledger.exists() else []
            indexed = {(r['attempt'],r.get('member')): r for r in previous}
            for row in rows:
                key = row['attempt'], row.get('member')
                if row['status'] == 'incomplete' and indexed.get(key, {}).get('status') not in (None,'incomplete'):
                    continue
                indexed[key] = row
            write_json(ledger, list(indexed.values()))
    publish()

    def assess(case, member):
        path = freeze_differential(root, case, schemas, member=member, engine=engine,
                                   engine_inventory=inventory, legacy=legacy)
        if path is None:
            return dict(status='incomplete', reason='frozen_member_unavailable')
        if resume:
            for previous in sorted(path.parent.glob('trial-*/summary.json')):
                saved = json.loads(previous.read_text())
                if saved.get('status') != 'incomplete':
                    return saved
        if deadline is not None and time.monotonic() >= deadline:
            return dict(status='incomplete', reason='budget_exhausted', manifest=str(path))
        return replay_devices(path, timeout=timeout, deadline=deadline, live=True, admitted=admitted or start)

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='device-pair') as pool:
        futures = {pool.submit(copy_context().run, assess, case, member): index
                   for index, case, member in tasks}
        for future in as_completed(futures):
            rows[futures[future]].update(future.result())
            publish()
    def paired(row):
        sides = row.get('devices', {})
        return all(sides.get('0-' + d, {}).get('target_returned') is True for d in ('cpu','cuda'))
    return dict(version=1 if legacy else 2, rows=rows,
                counts=dict(Counter(r['status'] for r in rows)), enabled=enabled,
                required_members=sum(r.get('legal_member') is True for r in rows),
                paired_target_returns=sum(paired(r) for r in rows),
                measured_members=sum(bool((r.get('comparison') or {}).get('raw', {}).get('fields')) for r in rows),
                differing_members=sum(bool((r.get('comparison') or {}).get('raw', {}).get('differences')) for r in rows),
                stable_candidates=sum(bool((r.get('comparison') or {}).get('stable_observed_fields')) for r in rows),
                comparable=sum(r['status'] in ('agreement','stable_mismatch','unstable_or_unconfirmed')
                               for r in rows),
                elapsed_seconds=time.monotonic()-start,
                worker_seconds=sum(proc['elapsed_seconds'] for r in rows for proc in r.get('processes', [])),
                incomplete=any(r['status'] == 'incomplete' for r in rows))


@differential_pool.standalone
def resume_check(root):
    from fcntl import flock, LOCK_EX, LOCK_NB
    root = Path(root)
    with (root/'resume.lock').open('a') as lock:
        flock(lock, LOCK_EX | LOCK_NB)
        report = json.loads((root/'summary.json').read_text())
        if report.get('version') != 'api-check-1':
            raise ValueError('Expected an api-check-1 run')
        pipeline = root/'execution/pipeline-state.json'
        native = root/'execution/summary.json'
        execution_complete = root/'execution-complete.json'
        recover = pipeline.exists() and (not execution_complete.exists() or
                  (pipeline.parent/'pipeline-pending-source.json').exists() or
                  report.get('native_stop_reason') in ('interrupted','pipeline_incomplete'))
        if report.get('terminal') and not recover and not report.get('differential', {}).get('incomplete'):
            return report
        if recover:
            from mirage.runtime.pipeline import resume_pipeline
            from mirage.artifacts.report import summarize_api_check
            restored = resume_pipeline(native.parent)
            report.update(stage='differential', terminal=False, metrics=summarize_api_check(restored),
                          native_stop_reason=restored['stop_reason'], generation_resumed=False)
            write_json(root/'summary.json', report)
            write_json(execution_complete, dict(stop_reason=restored['stop_reason'], records=len(restored['records']),
                                                generation_resumed=False))
        if not native.is_file() or not execution_complete.is_file():
            raise ValueError('No complete frozen generation summary available for provider-free resume')
        options = report['configuration']
        started = time.monotonic()
        # A pre-v2 frozen engine only understands the archived member comparator.
        frozen_policy = root/'execution/engine/mirage/inspector/comparison.py'
        legacy = frozen_policy.is_file() and 'PAIRED_DEVICE_POLICY' not in frozen_policy.read_text()
        report['differential'] = assess_devices(root, json.loads(native.read_text()),
            json.loads((root/'api-spec.json').read_text()), enabled=options['differential'],
            timeout=options['worker_timeout'], deadline=started+options.get('wall_limit_seconds', options['time_limit']),
            workers=options.get('workers',1), resume=True, legacy=legacy)
        report['resume_seconds'] = time.monotonic()-started
        from mirage.artifacts.report import check_completion
        completion, report['generation_completion'] = check_completion(
            json.loads(native.read_text()), report['differential'], (report.get('metrics') or {}).get('runtime_valid'))
        status, reason, exit_status = completion
        if report.get('generation_resumed') is False and status.startswith('completed'):
            status, reason = 'partial', 'submitted_work_drained_after_interruption'
        report.update(terminal=True, status=status, reason=reason, exit_status=exit_status)
        write_json(root/'summary.json', report)
        return report

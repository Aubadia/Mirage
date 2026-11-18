"""Run frozen evidence through isolated workers without model requests."""
from dataclasses import asdict
import json
from pathlib import Path
import time
import uuid

from mirage.artifacts.bundle import digest, verify
from mirage.runtime.process import run_worker
from mirage.workflow import replay, write_json

__all__ = ["replay"]


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


def freeze_differential(root, row, schemas):
    from mirage.inspector.comparison import DEVICE_POLICY
    directory = Path(row['directory'])
    engine = root/'execution/engine'
    # Representation children test the follow-up representation, including its mutation.
    name = 'followup' if row.get('proposal_origin') == 'framework_representation' else 'source'
    source, original = directory/'program.py', directory/'execution'/f'{name}.pt'
    if not original.is_file():
        return None
    dest = root/'differential'/f"attempt-{row['attempt']:04d}"
    dest.mkdir(parents=True, exist_ok=True)
    path = dest/'manifest.json'
    manifest = dict(format='device-differential-1', policy=DEVICE_POLICY, engine=str(engine),
                    engine_inventory={str(f.relative_to(engine)):digest(f) for f in sorted(engine.rglob('*.py'))},
                    source=str(source), original=str(original),
                    sha256={str(f): digest(f) for f in (source, original)}, schemas=schemas,
                    primary_verdict=row.get('mr_verdict'), primary_reason=row.get('reason'))
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError('Differential manifest changed')
        return path
    write_json(path, manifest)
    return path


def replay_devices(path, *, timeout=60, deadline=None):
    path = Path(path)
    manifest = json.loads(path.read_text())
    from mirage.inspector.comparison import DEVICE_POLICY
    if manifest['format'] != 'device-differential-1' or manifest['policy'] != DEVICE_POLICY:
        raise ValueError('Unknown device comparison policy')
    verify(Path(manifest['engine']), manifest['engine_inventory'])
    for filename, expected in manifest['sha256'].items():
        if digest(Path(filename)) != expected:
            raise ValueError('Differential evidence changed')
    root = path.parent/('trial-' + uuid.uuid4().hex)
    root.mkdir()
    report = dict(status='incomplete', directory=str(root), manifest=str(path), processes=[],
                  primary_verdict=manifest['primary_verdict'], primary_reason=manifest['primary_reason'],
                  provider_calls=0, confirmed_defect=False)
    def save(status, reason=None):
        report.update(status=status, reason=reason)
        write_json(root/'summary.json', report)
        return report
    save('incomplete', 'trial_started')
    def worker(label, request):
        remaining = timeout if deadline is None else min(timeout, deadline-time.monotonic())
        process = run_worker(dict(request, engine=manifest['engine']), root/label, max(0, remaining))
        report['processes'].append(asdict(process))
        save('incomplete', label)
        if process.outcome != 'completed':
            save('incomplete', label + '_' + process.outcome)
            return None
        return json.loads((root/label/'summary.json').read_text())
    members = []
    for round_index in range(2):
        for device in ('cpu', 'cuda'):
            label = f'{round_index}-{device}'
            result = worker(label, dict(mode='device_member', device=device, original=manifest['original'],
                source=Path(manifest['source']).read_text(), schemas=manifest['schemas']))
            if result is None:
                return report
            if result['status'] != 'ready':
                return save(result['status'], result['reason'])
            members.append(str(root/label/'member.pt'))
        comparison = worker(f'{round_index}-comparison', dict(mode='compare_devices', members=members))
        if comparison is None:
            return report
        report['comparison'] = comparison
        if comparison['status'] != 'mismatch' or round_index:
            return save(comparison['status'], comparison.get('reason'))
    raise RuntimeError('Device comparison ended without a terminal result')


def assess_devices(root, result, spec, *, enabled, timeout, deadline, resume=False):
    rows = []
    from collections import Counter
    started = time.monotonic()
    schemas = [item['schema'] for item in spec.get('operator_schemas', [])]
    for case in result.get('records', []):
        route = primary_route(case)
        row = dict(attempt=case['attempt'], claim=case.get('configuration', {}).get('check_claim'),
                   primary_route=route, primary_reason=case.get('reason'), status='not_requested')
        if route == 'fallback':
            if not enabled:
                row.update(status='disabled')
            elif not spec.get('environment', {}).get('cuda_available'):
                row.update(status='incomparable', reason='cuda_unavailable')
            elif row['claim'] not in ('forward', 'operand_relation'):
                row.update(status='incomparable', reason='claim_history_or_observables_unsupported')
            else:
                path = freeze_differential(root, case, schemas)
                previous = (json.loads(p.read_text()) for p in sorted(path.parent.glob('trial-*/summary.json'))) if path and resume else ()
                terminal = next((saved for saved in previous if saved['status'] != 'incomplete'), None)
                if terminal:
                    row.update(terminal)
                elif path is None:
                    row.update(status='incomplete', reason='frozen_member_unavailable')
                elif time.monotonic() >= deadline:
                    row.update(status='incomplete', reason='budget_exhausted', manifest=str(path))
                else:
                    row.update(replay_devices(path, timeout=timeout, deadline=deadline))
        rows.append(row)
        write_json(root/'differential-ledger.json', rows)
    return dict(rows=rows, counts=dict(Counter(r['status'] for r in rows)), enabled=enabled,
                elapsed_seconds=time.monotonic()-started,
                worker_seconds=sum(proc['elapsed_seconds'] for r in rows for proc in r.get('processes', [])),
                incomplete=any(r['status'] == 'incomplete' for r in rows))


def resume_check(root):
    from fcntl import flock, LOCK_EX, LOCK_NB
    root = Path(root)
    with (root/'resume.lock').open('a') as lock:
        flock(lock, LOCK_EX | LOCK_NB)
        report = json.loads((root/'summary.json').read_text())
        if report.get('version') != 'api-check-1':
            raise ValueError('Expected an api-check-1 run')
        if report.get('terminal') and not report.get('differential', {}).get('incomplete'):
            return report
        native = root/'execution/summary.json'
        if not native.is_file() or not (root/'execution-complete.json').is_file():
            raise ValueError('No complete frozen generation summary available for provider-free resume')
        options = report['configuration']
        started = time.monotonic()
        report['differential'] = assess_devices(root, json.loads(native.read_text()),
            json.loads((root/'api-spec.json').read_text()), enabled=options['differential'],
            timeout=options['worker_timeout'], deadline=started+options['time_limit'], resume=True)
        report['resume_seconds'] = time.monotonic()-started
        incomplete = report['differential']['incomplete']
        native_data = json.loads(native.read_text())
        if any(r.get('status') == 'provider_failed' or r.get('process', {}).get('outcome') in ('crash', 'timeout')
               for r in native_data.get('records', [])) or native_data.get('stop_reason') in (
                'metadata_failed', 'controller_error', 'provider_cancellation_unconfirmed'):
            report.update(terminal=True, exit_status=1, status='failed', reason='execution_incomplete')
        elif not (report.get('metrics') or {}).get('runtime_valid'):
            report.update(terminal=True, exit_status=0, status='skipped', reason='no_verified_execution')
        elif report.get('stage') == 'differential':
            report.update(terminal=True, exit_status=1 if incomplete else 0,
                          status='partial' if incomplete else 'completed',
                          reason='differential_incomplete' if incomplete else 'bounded_execution_finished')
        write_json(root/'summary.json', report)
        return report

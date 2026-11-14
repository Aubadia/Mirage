"""API-independent routing and scoped corroboration of numerical evidence."""
import time

AUDIT_VERSION = 'numerical-gradient-1'


def unavailable(reason, *, status='unknown'):
    return {'version': AUDIT_VERSION, 'status': status, 'scope': 'target_vjp',
            'reason': reason, 'events': [], 'cost': {'seconds': 0.0}}


def audit_execution(record, *, budget_seconds=10):
    """Run an auxiliary check only after the original execution is complete."""
    if not record.get('completed') or not record.get('inspection', {}).get('valid'):
        return unavailable('original execution/state/path is not verified')
    if not record.get('gradient_evidence'):
        return unavailable('legacy record has no numerical backward evidence')
    if record.get('run_rng_unchanged') is not True:
        return unavailable('execution RNG effects are not certified', status='unsupported')
    if not record.get('program_certificate'):
        return unavailable('source effects are outside the existing execution certificate', status='unsupported')
    started = time.monotonic()
    try:
        from mirage.inspector.numerical_gradients import audit_record
        result = audit_record(record, budget_seconds=budget_seconds)
    except (ValueError, TypeError, KeyError, RuntimeError, NotImplementedError) as error:
        result = unavailable(f'audit could not complete: {type(error).__name__}: {error}')
    result.setdefault('cost', {})['seconds'] = time.monotonic() - started
    return result


def verified_violation_calls(record):
    """Never attribute a violation at one call to a trigger at another call."""
    audit = record.get('gradient_audit', {})
    evidence = record.get('gradient_evidence', {})
    if (audit.get('version') != AUDIT_VERSION or audit.get('scope') != 'target_vjp'
            or audit.get('status') != 'violation'
            or evidence.get('version') != 'gradient-evidence-1'
            or not record.get('completed') or not record.get('inspection', {}).get('valid')):
        return ()
    completed = {item['index'] for item in record.get('backward_invocations', []) if item.get('completed')}
    references = {(item.get('call'), item.get('node'), item.get('invocation'))
                  for item in evidence.get('events', []) if item.get('status') == 'complete'
                  and item.get('invocation') in completed}
    calls = {item['index'] for item in record.get('target_calls', [])}
    return tuple(sorted({item['call'] for item in audit.get('events', [])
                         if item.get('status') == 'violation' and item.get('baseline_verified') is True
                         and item.get('scope') == 'target_vjp'
                         and item.get('stable_steps') is True and item.get('one_sided_consistent') is True
                         and any(d.get('status') == 'violation' and d.get('stable_steps') is True
                                 and d.get('one_sided_consistent') is True for d in item.get('directions', []))
                         and (item.get('call'), item.get('node'), item.get('invocation')) in references
                         and item.get('call') in calls}))


def audit_pair(left, right, summary, *, budget_seconds=10, enabled=True):
    """Keep MR verdicts and numerical agreement distinct; never synthesize MR Pass."""
    summary['mr_verdict'] = summary.get('verdict')
    results = {}
    violated = []
    for member, record in (('source', left), ('followup', right)):
        if record is None:
            continue
        result = audit_execution(record, budget_seconds=budget_seconds) if enabled else unavailable('not selected by bounded audit schedule')
        record['gradient_audit'] = result
        results[member] = result
        if verified_violation_calls(record):
            violated.append(member)
    summary['gradient_audits'] = results
    summary['numerical_violation_members'] = violated
    if violated:
        summary.update(verdict='suspected_failure', correctness='assessed',
                       reason='Numerical directional derivative relation violated; replay/corroboration required')
    return results

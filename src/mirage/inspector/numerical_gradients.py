"""API-independent, representation-preserving directional gradient consistency.

No derivative formula or issue/API-name dispatch is used here. Agreement only
covers observed cotangents and tested directions; forward semantics are unproven.
"""
import math
import random
import time

from mirage.runtime.api_metadata import resolve
from mirage.runtime.representations import ReconstructionError, StorageCoordinateChart

VERSION = 'numerical-gradient-1'


class AuditUnknown(ValueError):
    pass


def _values(tensor):
    return [float(v) for v in tensor.detach().resolve_neg().cpu().reshape(-1).tolist()]


def _same_outputs(actual, saved):
    import torch
    if isinstance(saved, torch.Tensor):
        return (isinstance(actual, torch.Tensor) and actual.shape == saved.shape and actual.dtype == saved.dtype
                and bool(torch.isfinite(actual).all()) and torch.equal(actual.detach().cpu(), saved))
    if isinstance(saved, dict):
        return isinstance(actual, dict) and set(actual) == set(saved) and all(_same_outputs(actual[k], v) for k, v in saved.items())
    if isinstance(saved, (list, tuple)):
        return isinstance(actual, (list, tuple)) and len(actual) == len(saved) and all(_same_outputs(a, s) for a, s in zip(actual, saved))
    return type(actual) is type(saved) and actual == saved


def _output_seeds(output, event):
    import torch
    from mirage.runtime.program import leaves
    tensors = dict(leaves(output))
    seeds, selected, paths, slots = [], [], set(), set()
    for item in event['outputs']:
        path, slot = item['path'], item['slot']
        if path in paths or slot in slots or path not in tensors:
            raise AuditUnknown('ambiguous or missing output slot/path')
        paths.add(path); slots.add(slot)
        tensor, gradient = tensors[path], item.get('gradient')
        if gradient is None:
            continue
        if (not isinstance(gradient, torch.Tensor) or gradient.shape != tensor.shape or
                gradient.dtype != tensor.dtype or tensor.dtype not in (torch.float32, torch.float64) or
                not bool(torch.isfinite(gradient).all())):
            raise AuditUnknown('unsupported output cotangent')
        selected.append(tensor)
        seeds.append(gradient.detach().to(device=tensor.device))
    if not selected or not any(bool(seed.ne(0).any()) for seed in seeds):
        raise AuditUnknown('no informative output cotangent')
    return selected, seeds


def _scalar(output, event):
    selected, seeds = _output_seeds(output, event)
    # Python accumulation avoids using the target framework's reduction as the reference.
    terms = [a * b for tensor, seed in zip(selected, seeds)
             for a, b in zip(_values(tensor), _values(seed))]
    result = math.fsum(terms)
    mass = math.fsum(abs(term) for term in terms)
    if not math.isfinite(result) or not math.isfinite(mass):
        raise AuditUnknown('nonfinite projected output or magnitude')
    return result, mass


def _directions(size, maximum, seed):
    if size <= 16:
        return [[float(i == j) for i in range(size)] for j in range(size)]
    rng = random.Random(seed)
    result = []
    for _ in range(maximum):
        direction = [rng.uniform(-1., 1.) for _ in range(size)]
        norm = math.sqrt(math.fsum(v * v for v in direction))
        result.append([v / norm for v in direction])
    return result


def _audit_event(record, call, event, target, deadline, cost, *, max_directions, max_coordinates, max_elements):
    import torch
    chart = StorageCoordinateChart(call, max_coordinates=max_coordinates, max_elements=max_elements)
    expected = chart.pullback(event['inputs'])
    context = call.get('entry_context', {})
    if not context.get('grad_enabled') or any(context.get(k) for k in ('cpu_enabled', 'cuda_enabled', 'inference_enabled')):
        raise AuditUnknown('unsupported grad/autocast/inference entry context')
    devices = sorted({torch.device(g['device']).index if torch.device(g['device']).index is not None else torch.cuda.current_device()
                      for g in chart.groups.values() if torch.device(g['device']).type == 'cuda'})

    def evaluate(values, *, verify=False):
        if time.monotonic() >= deadline:
            raise AuditUnknown('numerical audit time budget exhausted')
        tree, bases = chart.build(values)
        if verify:
            chart.verify(tree)
        tensors = []
        from mirage.runtime.program import leaves
        for _, tensor in leaves(tree):
            tensors.append((tensor, tensor._version))
        before_cpu = torch.get_rng_state()
        before_cuda = {device: torch.cuda.get_rng_state(device) for device in devices}
        try:
            output = target(*tree['args'], **tree['kwargs'])
            cost['forward_evaluations'] += 1
            if sum(t.numel() for _, t in leaves(output)) > max_elements:
                raise AuditUnknown('output element budget exceeded')
            if not torch.equal(before_cpu, torch.get_rng_state()) or any(
                    not torch.equal(state, torch.cuda.get_rng_state(device)) for device, state in before_cuda.items()):
                raise AuditUnknown('target changes RNG state')
            if any(tensor._version != version for tensor, version in tensors):
                raise AuditUnknown('target mutates reconstructed arguments')
            return output, bases
        finally:
            torch.set_rng_state(before_cpu)
            for device, state in before_cuda.items():
                torch.cuda.set_rng_state(state, device)

    with torch.enable_grad():
        baseline, bases = evaluate(chart.values, verify=True)
        if not _same_outputs(baseline, call.get('output')):
            raise AuditUnknown('reconstructed baseline output differs from original')
        repeated, _ = evaluate(chart.values, verify=True)
        if not _same_outputs(repeated, call.get('output')):
            raise AuditUnknown('baseline forward is not reproducible')
        selected, seeds = _output_seeds(baseline, event)
        if not all(t.requires_grad for t in selected):
            raise AuditUnknown('reconstructed outputs lack observed gradient connection')
        actual = torch.autograd.grad(selected, [base for _, base in bases], grad_outputs=seeds, allow_unused=True)
        cost['backward_evaluations'] += 1
        if any(gradient is None for gradient in actual):
            raise AuditUnknown('reconstructed VJP has missing storage coordinates')
        actual_values = [v for gradient in actual for v in _values(gradient)]
        epsilon = chart.epsilon
        for a, e in zip(actual_values, expected):
            if not math.isfinite(a) or abs(a-e) > 64 * epsilon * max(abs(a), abs(e), 1e-12):
                raise AuditUnknown('reconstructed VJP differs from observed original VJP')
        if len(actual_values) != len(expected):
            raise AuditUnknown('reconstructed VJP coordinate count differs')
        phi0, mass0 = _scalar(baseline, event)
        results = []
        values = chart.values
        # Bound absolute displacement too: a large unrelated coordinate must not
        # turn a local derivative probe into a large-domain transformation.
        h = min(.01, epsilon ** (1 / 3) * max(1., max(abs(v) for v in values)))
        for index, direction in enumerate(_directions(len(values), max_directions,
                                                      7919 * event['call'] + 101 * event['invocation'] + event['node'])):
            predicted = math.fsum(v*g for v, g in zip(direction, expected))
            if not math.isfinite(predicted):
                raise AuditUnknown('nonfinite projected VJP')
            samples = []
            for multiplier in (.5, 1., 2.):
                step = h * multiplier
                plus, plus_bases = evaluate([z+step*v for z, v in zip(values, direction)])
                minus, minus_bases = evaluate([z-step*v for z, v in zip(values, direction)])
                rounding = []
                for sign, reconstructed in ((1, plus_bases), (-1, minus_bases)):
                    cast = [v for _, base in reconstructed for v in _values(base)]
                    for z, v, actual_coordinate in zip(values, direction, cast):
                        requested = sign * step * v
                        if v and actual_coordinate == z:
                            raise AuditUnknown('requested perturbation rounds to unchanged coordinate')
                        rounding.append((actual_coordinate-z)-requested)
                relative_error = math.sqrt(math.fsum(e*e for e in rounding)) / (step*math.sqrt(2))
                if not math.isfinite(relative_error) or relative_error > 64 * epsilon ** (2/3):
                    raise AuditUnknown('dtype rounding changes perturbation direction materially')
                fp, massp = _scalar(plus, event)
                fm, massm = _scalar(minus, event)
                central = (fp-fm)/(2*step)
                forward, backward = (fp-phi0)/step, (phi0-fm)/step
                noise = 8*epsilon*max(massp,massm,mass0,1e-30)/step
                if not all(math.isfinite(v) for v in (central, forward, backward, noise)):
                    raise AuditUnknown('nonfinite finite-difference estimate or roundoff bound')
                samples.append({'step': step, 'central': central, 'right': forward, 'left': backward,
                                'roundoff_estimate': noise, 'direction_rounding_error': relative_error})
            central = [s['central'] for s in samples]
            spread = max(central)-min(central)
            magnitude = max(abs(predicted), *(abs(c) for c in central), 1e-12)
            base_tolerance = 64 * epsilon ** (2/3) * magnitude
            noise = max(s['roundoff_estimate'] for s in samples)
            gaps = [abs(s['right']-s['left']) for s in samples]
            nonsmooth = gaps[0] > max(8*noise, base_tolerance) and gaps[0] > .8*gaps[-1]
            stable = spread <= max(base_tolerance, 8*noise, .02*magnitude)
            limit = max(base_tolerance, 4*noise, 4*spread)
            errors = [abs(c-predicted) for c in central]
            if not all(math.isfinite(v) for v in (spread, magnitude, base_tolerance, limit, *gaps, *errors)):
                raise AuditUnknown('nonfinite finite-difference comparison')
            status = ('unknown' if nonsmooth or not stable else 'agreement' if max(errors) <= limit else
                      'violation' if min(errors) > 8*limit else 'unknown')
            results.append({'direction': index, 'vector': direction, 'vjp_projection': predicted,
                'samples': samples, 'comparison_limit': limit, 'status': status,
                'stable_steps': stable, 'one_sided_consistent': not nonsmooth,
                'reason': 'one-sided slopes do not converge' if nonsmooth else
                    'step sensitivity' if not stable else 'numerical uncertainty band' if status == 'unknown' else
                    'directional consistency' if status == 'agreement' else 'stable directional mismatch'})
        status = ('violation' if any(r['status']=='violation' for r in results) else
                  'agreement' if results and all(r['status']=='agreement' for r in results) else 'unknown')
        return {'call':event['call'],'node':event['node'],'invocation':event['invocation'],
            'status':status,'baseline_verified':True,'scope':'target_vjp',
            'reason':'observed VJP compared with reconstructed directional forward differences',
            'independent_coordinates':len(values),'directions':results,
            'coordinate_chart':'occupied physical storage values; signed shared-index pullback',
            'leaf_history_scope':'local call chart; original pre-target history is not replayed',
            'stable_steps':all(r['stable_steps'] for r in results if r['status']=='violation'),
            'one_sided_consistent':all(r['one_sided_consistent'] for r in results if r['status']=='violation')}


def audit_record(record, *, budget_seconds=30, max_directions=4, max_coordinates=512, max_elements=4096):
    """Return bounded JSON evidence; unsupported cases never count as a pass.

    Up to 16 coordinates use the full basis. ``max_directions`` bounds random
    directions for larger charts; the time budget applies to both cases.
    """
    import torch
    started = time.monotonic()
    cost = {'forward_evaluations':0,'backward_evaluations':0}
    report = {'version':VERSION,'status':'unknown','scope':'target_vjp','reason':'no supported numerical evidence',
              'events':[],'cost':cost,'limitations':[],
              'interpretation':'Scoped forward/backward consistency, not forward semantic correctness or a universal derivative certificate.'}
    def finish(status, reason):
        report.update(status=status, reason=reason)
        cost['elapsed_seconds'] = time.monotonic()-started
        return report
    if budget_seconds <= 0 or max_directions <= 0:
        return finish('unknown','numerical audit budget unavailable')
    if not record.get('completed') or not record.get('api_executed'):
        return finish('unknown','incomplete or unobserved target execution')
    evidence = record.get('gradient_evidence', {})
    if evidence.get('version') != 'gradient-evidence-1':
        return finish('unsupported','missing supported gradient evidence version')
    if record.get('run_rng_unchanged') is not True:
        return finish('unsupported','run RNG stability is not established')
    if torch.is_inference_mode_enabled() or torch.is_autocast_enabled('cpu') or torch.is_autocast_enabled('cuda'):
        return finish('unsupported','auditor requires ordinary grad context without autocast')
    try:
        target = resolve(record['config']['api'])
    except (KeyError, ValueError, AttributeError) as error:
        return finish('unsupported',str(error))
    completed = {item['index'] for item in record.get('backward_invocations',[]) if item.get('completed')}
    calls = {call['index']:call for call in record.get('target_calls',[])}
    events = evidence.get('events',[])
    report['limitations'] = evidence.get('limitations',[])[:16]
    identities = [(e.get('call'), e.get('node'), e.get('invocation')) for e in events]
    for event in events[:16]:
        identity = {key:event.get(key) for key in ('call','node','invocation')}
        try:
            call = calls.get(event.get('call'))
            if identities.count(tuple(identity.values())) != 1:
                raise AuditUnknown('duplicate or reentrant backward identity')
            if event.get('status') != 'complete' or event.get('invocation') not in completed:
                raise AuditUnknown('incomplete or ambiguous observed backward event')
            if call is None or call.get('execution_mode') != 'eager' or call.get('evidence'):
                raise AuditUnknown('local eager target boundary is unavailable')
            if call.get('schema_read_only') is not True or call.get('writable_paths') or call.get('write_effects'):
                raise AuditUnknown('read-only target effect certificate is unavailable')
            result = _audit_event(record, call, event, target, started+budget_seconds, cost,
                max_directions=max_directions,max_coordinates=max_coordinates,max_elements=max_elements)
        except (AuditUnknown, ReconstructionError, ValueError, TypeError, KeyError, IndexError, RuntimeError, OverflowError) as error:
            result = dict(identity,status='unknown',baseline_verified=False,scope='target_vjp',reason=str(error))
        report['events'].append(result)
        if time.monotonic() >= started+budget_seconds:
            report['limitations'].append({'reason':'time budget exhausted; remaining events unassessed'})
            break
    if any(e['status']=='violation' for e in report['events']):
        return finish('violation','at least one baseline-verified stable directional VJP mismatch')
    all_observed = len(events)==len(report['events']) and not evidence.get('dropped_events') and not evidence.get('dropped_limitations')
    if (report['events'] and all_observed and not report['limitations'] and
            all(e['status']=='agreement' for e in report['events'])):
        return finish('agreement','agreement for observed cotangents and tested directions only')
    return finish('unknown','some numerical evidence or stability conditions remain unresolved')

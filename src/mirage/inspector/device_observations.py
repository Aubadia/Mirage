"""Device observations at aligned public target boundaries, independently of oracle eligibility."""
import math
import torch

from mirage.backends.pytorch import PyTorchBackend


def exact(a, b):
    return PyTorchBackend().compare(a, b, 'exact')['equal'] is True


def metadata(items):
    return [{k: v for k, v in item.items() if k != 'device'} for item in items]


def call_gap(a, b):
    for key in ('index', 'execution_mode', 'entry_context', 'writable_paths'):
        if key not in a or a[key] != b.get(key):
            return 'call_' + key + '_changed'
    if metadata(a['metadata']) != metadata(b['metadata']):
        return 'input_metadata_or_alias_changed'
    if not exact(a['inputs'], b['inputs']):
        return 'semantic_input_changed'
    return None


def tensor_delta(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return dict(status='different', kind='tensor_contract', left=[list(a.shape), str(a.dtype)],
                    right=[list(b.shape), str(b.dtype)])
    if a.is_complex():
        return None  # components are observed separately by the caller
    mask = a != b
    floating = a.is_floating_point()
    if floating:
        mask &= ~(torch.isnan(a) & torch.isnan(b))
    count = int(mask.sum())
    result = dict(status='different' if count else 'equal', elements=a.numel(), unequal=count)
    if count:
        result['coordinates'] = mask.nonzero()[:8].tolist()
        # Record representative values losslessly, including integers larger than 2**53.
        result['examples'] = [[str(a[tuple(i)].item()), str(b[tuple(i)].item())] for i in result['coordinates']]
    if floating:
        for name, op in [('nan',torch.isnan), ('posinf',torch.isposinf), ('neginf',torch.isneginf)]:
            result[name + '_position_changes'] = int((op(a) != op(b)).sum())
        finite = torch.isfinite(a) & torch.isfinite(b)
        if finite.any():
            x, y = a[finite].double(), b[finite].double()
            absolute = (x-y).abs().max().item()
            scale = torch.maximum(x.abs(), y.abs())
            relative = ((x / scale.where(scale != 0, 1)) - (y / scale.where(scale != 0, 1))).abs().max().item()
            result.update(max_absolute=absolute if math.isfinite(absolute) else 'overflow',
                          max_symmetric_relative=relative)
    return result


def observe_pair(left, right, original=None):
    fields, gaps = {}, {}

    def walk(path, a, b):
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            value = tensor_delta(a, b)
            if value is not None:
                fields[path] = value
            else:
                walk(path + '/real', a.real, b.real)
                walk(path + '/imag', a.imag, b.imag)
        elif type(a) != type(b):
            fields[path] = dict(status='different', kind='type', left=type(a).__name__, right=type(b).__name__)
        elif isinstance(a, (list, tuple, dict)):
            if isinstance(a, dict) and (set(a) | set(b)) & {'opaque_type','callable','stateful_type'}:
                gaps[path] = 'opaque_value'
                return
            keys_a = set(a) if isinstance(a, dict) else set(range(len(a)))
            keys_b = set(b) if isinstance(b, dict) else set(range(len(b)))
            if keys_a != keys_b:
                fields[path] = dict(status='different', kind='structure', left=sorted(keys_a), right=sorted(keys_b))
            elif not keys_a:
                fields[path] = dict(status='equal', kind='empty_structure')
            for key in sorted(keys_a & keys_b):
                walk(f'{path}/{key!r}', a[key], b[key])
        elif type(a) in (str, int, bool, float, type(None)):
            same = a == b or (type(a) == float and math.isnan(a) and math.isnan(b))
            fields[path] = dict(status='equal' if same else 'different', left=str(a), right=str(b))
        else:
            gaps[path] = 'unrecorded_value_type'

    records = [left, right] if original is None else [original, left, right]
    reference = records[0]
    for rec in records[1:]:
        same = rec['config']['api'] == reference['config']['api'] and all(
            rec['environment'].get(k) == reference['environment'].get(k)
            for k in ('torch','cuda_runtime','deterministic_algorithms','matmul_precision','mirage_code_sha256'))
        if not same or not exact(rec['initial'], reference['initial']) or (
                metadata(rec['initial_metadata']) != metadata(reference['initial_metadata'])):
            return dict(fields={}, gaps={'case':'environment_or_initial_state_changed'}, differences=[])
    for i in range(max(len(r['target_calls']) for r in records)):
        path = f'call/{i}'
        if any(len(r['target_calls']) <= i for r in records):
            gaps[path] = 'call_not_recorded'
            continue
        calls = [r['target_calls'][i] for r in records]
        reason = next((g for c in calls[1:] if (g := call_gap(calls[0], c))), None)
        if reason:
            gaps[path] = reason
            continue
        a, b = calls[-2:]
        if 'target_exception' in a or 'target_exception' in b:
            walk(path+'/completion', a.get('target_exception','returned'), b.get('target_exception','returned'))
            continue
        if 'output_structure' in a and 'output_structure' in b and a['output_structure'] != b['output_structure']:
            fields[path+'/structure'] = dict(status='different',kind='container',
                                              left=a['output_structure'],right=b['output_structure'])
        elif 'output_structure' in a and 'output_structure' in b and isinstance(a['output_structure'], list):
            fields[path+'/structure'] = dict(status='equal',kind='container')
        walk(path+'/output', a['output'], b['output'])
        if a.get('writable_paths'):
            walk(path+'/writes', a['destination_values'], b['destination_values'])
        if a.get('guard_preserved') is not None or b.get('guard_preserved') is not None:
            walk(path+'/guard', a.get('guard_preserved'), b.get('guard_preserved'))
        events = [[e for e in r.get('gradient_evidence', {}).get('events', []) if e.get('call') == i] for r in records]
        if any(events):
            identities = [[(e.get('node'),e.get('invocation')) for e in es] for es in events]
            histories = [r.get('backward_invocations', []) for r in records]
            if any(ids != identities[0] for ids in identities) or any(h != histories[0] for h in histories):
                gaps[path+'/vjp'] = 'backward_history_changed'
                continue
            for j, group in enumerate(zip(*events)):
                vp = f'{path}/vjp/{j}'
                if (any(e.get('status') != 'complete' or not e.get('operands_unchanged_at_backward')
                        or not e.get('outputs') for e in group) or any(
                        not exact(group[0]['outputs'], e['outputs']) for e in group[1:])):
                    gaps[vp] = 'cotangent_or_backward_evidence_unaligned'
                    continue
                x, y = group[-2:]
                if [s['path'] for s in x['inputs']] != [s['path'] for s in y['inputs']]:
                    gaps[vp] = 'gradient_coordinates_changed'
                    continue
                for lhs, rhs in zip(x['inputs'],y['inputs']):
                    walk(f"{vp}/{lhs['path']}", lhs['gradient'], rhs['gradient'])
    if any(r.get('context_events') for r in records):
        histories = [r.get('context_events', []) for r in records]
        history_keys = ('factory_call','instance_group','entry_depth','decorated_entry','before')
        if any(len(h) != len(histories[0]) or any(not event.get('balanced') for event in h)
               for h in histories):
            gaps['context'] = 'scope_history_unaligned'
        else:
            for i, group in enumerate(zip(*histories)):
                if all(all(e.get(k) == group[0].get(k) for k in history_keys) for e in group):
                    walk(f'context/{i}/exit',group[-2].get('after'),group[-1].get('after'))
                else:
                    gaps[f'context/{i}'] = 'scope_entry_unaligned'
    uncertain_rng = any(r.get('initialization_rng_changed') or r.get('run_rng_unchanged') is not True
                        for r in records)
    return dict(fields=fields, history_rng_unverified=uncertain_rng,
                gaps=gaps or ({} if fields else {'case':'no_materialized_observable'}),
                differences=[k for k,v in fields.items() if v['status']=='different'])

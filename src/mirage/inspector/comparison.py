"""Recursive numerical comparison with explicit finite-value and dtype checks."""

import math
from pathlib import Path


def differences(left, right, tolerance: dict | None = None, path: str = "value") -> list[dict]:
    import torch

    tolerance = tolerance or {"atol": 0.0, "rtol": 0.0}
    if path.endswith(".step") or ".optimizer.param_groups" in path or path.startswith("optimizer.param_groups"):
        tolerance = {"atol": 0.0, "rtol": 0.0}
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            return [{"path": path, "reason": "tensor/type mismatch"}]
        if left.shape != right.shape or left.dtype != right.dtype:
            return [{"path": path, "reason": "shape/dtype mismatch"}]
        if left.is_floating_point() or left.is_complex():
            if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
                return [{"path": path, "reason": "nonfinite tensor"}]
            dtype = torch.complex128 if left.is_complex() else torch.float64
            a, b = left.to(dtype), right.to(dtype)
            error = (a - b).abs()
            allowed = tolerance["atol"] + tolerance["rtol"] * b.abs()
            if bool((error <= allowed).all()):
                return []
            return [{"path": path, "reason": "numerical mismatch",
                     "max_absolute_error": float(error.max()), "tolerance": tolerance}]
        return [] if torch.equal(left, right) else [{"path": path, "reason": "discrete tensor mismatch"}]
    if type(left) is not type(right):
        return [{"path": path, "reason": "type mismatch"}]
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return [{"path": path, "reason": "mapping keys mismatch"}]
        return [item for key in left for item in differences(left[key], right[key], tolerance, f"{path}.{key}")]
    if isinstance(left, (tuple, list)):
        if len(left) != len(right):
            return [{"path": path, "reason": "sequence length mismatch"}]
        return [item for index, (a, b) in enumerate(zip(left, right))
                for item in differences(a, b, tolerance, f"{path}[{index}]")]
    if isinstance(left, float):
        if not math.isfinite(left) or not math.isfinite(right):
            return [{"path": path, "reason": "nonfinite scalar"}]
        if abs(left - right) <= tolerance["atol"] + tolerance["rtol"] * abs(right):
            return []
    elif left == right:
        return []
    return [{"path": path, "reason": "scalar mismatch", "left": left, "right": right}]


def equal(left, right) -> bool:
    return not differences(left, right)



DEVICE_POLICY = 'eager-pointwise-v1'


def device_admission(record, schemas):
    """One direct, deterministic, finite, single-valued operation on its declared device."""
    import torch
    from mirage.backends.pytorch import _REPRESENTATION_SEMANTIC_CLASSES
    config = record['config']
    if not record['inspection']['valid'] or config.get('check_claim') not in ('forward', 'operand_relation'):
        return 'claim_history_or_observables_unsupported'
    if config.get('requires_backward') or record.get('backward_invocations'):
        return 'cotangent_alignment_unavailable'
    if record.get('initialization_rng_changed') or record.get('run_rng_unchanged') is not True:
        return 'random_inputs_or_history'
    calls = record.get('target_calls', [])
    if len(calls) != 1 or calls[0].get('execution_mode') != 'eager':
        return 'single_eager_call_required'
    call = calls[0]
    native = call.get('native_dispatch', [])
    if len(native) != 1 or native[0]['schema'] not in schemas:
        return 'exact_native_overload_unavailable'
    tags = native[0]['tags']
    op = native[0]['schema'].split('(')[0].split('::')[-1].split('.')[0]
    if 'pointwise' not in tags or any('nondeterministic' in tag for tag in tags):
        return 'single_valued_pointwise_policy_unavailable'
    if not call.get('schema_read_only') or call.get('write_effects'):
        return 'write_or_alias_effect_policy_unavailable'
    context = call['entry_context']
    if context['cpu_enabled'] or context['cuda_enabled'] or context['inference_enabled']:
        return 'precision_or_execution_context_unsupported'
    tensors = call.get('metadata', [])
    device = config['device'].split(':')[0]
    if any(t['device'].split(':')[0] != device for t in record['initial_metadata']):
        return 'mixed_initial_device_placement'
    if not tensors or any(t['device'].split(':')[0] != device or t['layout'] != 'torch.strided'
                          or t['storage_group'] is None for t in tensors):
        return 'operand_placement_or_representation_unsupported'
    output = call['output']
    if not isinstance(output, torch.Tensor) or not output.numel() or not torch.isfinite(output).all():
        return 'output_domain_unsupported'
    if any(t['properties']['complex'] for t in tensors) or output.is_complex():
        return 'complex_domain_unsupported'
    if output.is_floating_point() or any(t['properties']['floating'] for t in tensors):
        if output.dtype not in (torch.float32, torch.float64) or op not in _REPRESENTATION_SEMANTIC_CLASSES['pointwise']:
            return 'floating_point_policy_unavailable'
        from mirage.runtime.program import leaves
        for _, x in leaves(call['inputs']):
            if x.dtype != output.dtype or not x.numel() or not torch.isfinite(x).all():
                return 'input_domain_unsupported'
            if not ((x.abs() >= 0.125) & (x.abs() <= 8)).all():
                return 'outside_frozen_numerical_domain'
    elif not op.startswith(('bitwise_and', 'bitwise_or', 'bitwise_xor', 'bitwise_not',
                            'logical_and', 'logical_or', 'logical_xor', 'logical_not')):
        return 'discrete_semantics_unreviewed'
    return None


def aligned_device_records(original, candidate):
    """Require the frozen logical state, schema, alias partitions and target checkpoint."""
    from mirage.backends.pytorch import PyTorchBackend
    def equal(a, b):
        return PyTorchBackend().compare(a, b, 'exact')['equal'] is True
    def descriptors(items):
        return [{k: v for k, v in item.items() if k != 'device'} for item in items]
    a, b = original['target_calls'][0], candidate['target_calls'][0]
    for key in ('torch', 'cuda_runtime', 'deterministic_algorithms', 'matmul_precision', 'mirage_code_sha256'):
        if original['environment'][key] != candidate['environment'][key]:
            return 'environment_changed'
    if (not equal(original['initial'], candidate['initial']) or
            descriptors(original['initial_metadata']) != descriptors(candidate['initial_metadata'])):
        return 'initial_state_or_layout_changed'
    if a['native_dispatch'] != b['native_dispatch'] or a['entry_context'] != b['entry_context']:
        return 'overload_or_context_changed'
    if descriptors(a['metadata']) != descriptors(b['metadata']) or not equal(a['inputs'], b['inputs']):
        return 'target_inputs_or_alias_topology_changed'
    if (original['transition_signature'] != candidate['transition_signature'] or
            [c['name'] for c in original['checkpoints']] != [c['name'] for c in candidate['checkpoints']]):
        return 'semantic_checkpoint_changed'
    return None


def device_comparison(left, right):
    import torch
    if left.dtype != right.dtype or left.shape != right.shape:
        return dict(status='incomparable', reason='output_contract_changed')
    if not (torch.isfinite(left).all() and torch.isfinite(right).all()):
        return dict(status='incomparable', reason='nonfinite_output')
    if not left.is_floating_point():
        return dict(status='agreement' if torch.equal(left, right) else 'mismatch',
                    policy=DEVICE_POLICY, unequal_elements=int((left != right).sum()), tolerance='exact')
    a, b = left.double(), right.double()
    delta = (a - b).abs()
    eps = torch.finfo(left.dtype).eps
    limit = 8 * eps * torch.maximum(a.abs(), b.abs()).clamp_min(1)
    return dict(status='agreement' if (delta <= limit).all() else 'mismatch', policy=DEVICE_POLICY,
                max_abs_delta=float(delta.max()), unequal_elements=int((delta > limit).sum()),
                tolerance='8 eps * max(1, abs(cpu), abs(cuda))', epsilon=eps)


def differential_worker(request, directory):
    import torch
    from mirage.runtime.stateful import execute
    from mirage.runtime.probes import Recorder
    from mirage.artifacts.bundle import load_tensor_record
    original = load_tensor_record(Path(request['original']))
    reason = device_admission(original, request['schemas'])
    if reason:
        return dict(status='incomparable', reason=reason)
    device = request['device']
    if device == 'cuda' and not torch.cuda.is_available():
        return dict(status='incomparable', reason='cuda_unavailable')
    try:
        record = execute(request['source'], dict(original['config'], device=device), Recorder(directory/'trace.jsonl'))
    except NotImplementedError as error:
        return dict(status='incomparable', reason='backend_operation_unavailable', diagnostic=str(error))
    if device == 'cuda':
        torch.cuda.synchronize()
    reason = device_admission(record, request['schemas']) or aligned_device_records(original, record)
    torch.save(record, directory/'member.pt')
    return dict(status='incomparable' if reason else 'ready', reason=reason, device=device,
                synchronized=device == 'cuda', environment=record['environment'])


def compare_saved_devices(request):
    from mirage.artifacts.bundle import load_tensor_record
    values = [load_tensor_record(Path(path))['target_calls'][0]['output'] for path in request['members']]
    first = device_comparison(*values[:2])
    if len(values) == 4:
        second = device_comparison(*values[2:])
        within = [device_comparison(values[i], values[i + 2]) for i in (0, 1)]
        stable = first['status'] == second['status'] == 'mismatch' and all(x['status'] == 'agreement' for x in within)
        return dict(status='stable_mismatch' if stable else 'unstable_or_unconfirmed',
                    first=first, repeat=second, within_device=within, confirmed_defect=False)
    return first

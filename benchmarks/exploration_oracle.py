"""Frozen four-arm traversal/oracle experiment; one process per search.

Lock is metadata-only. Run and reassess require a Slurm step. An interrupted
search is never retried; resumption skips terminal searches and preserves failures.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

APIS = ('torch.sin', 'torch.tanh', 'torch.add', 'torch.mul', 'torch.mm',
        'torch.bmm', 'torch.nn.functional.linear', 'torch.linalg.slogdet')
ARMS = {'A': (False, False), 'B': (True, False), 'C': (False, True), 'D': (True, True)}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.pending')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root):
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*.py'))
            if '__pycache__' not in p.parts}


def utc():
    return datetime.now(timezone.utc).isoformat()


def require_allocation():
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Computational phases require an allocated Slurm step')


def locked_protocol(path):
    protocol = json.loads(path.read_text())
    for name, value in protocol['engines'].items():
        if inventory(Path(value['path'])) != value['sha256']:
            raise RuntimeError(f'Frozen {name} engine changed')
    if inventory(Path(protocol['controls_path'])) != protocol['controls_sha256']:
        raise RuntimeError('Frozen control sources changed')
    if digest(Path(protocol['driver'])) != protocol['driver_sha256']:
        raise RuntimeError('Frozen experiment driver changed')
    return protocol


def lock(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    engines = {}
    for label, source in (('baseline', args.baseline), ('candidate', args.candidate)):
        destination = output / 'engines' / label / 'src'
        shutil.copytree(source.resolve(), destination,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        engines[label] = {'path': str(destination), 'sha256': inventory(destination)}
    controls = output / 'controls'
    shutil.copytree(Path(__file__).resolve().parent / 'pytorch_correctness', controls / 'pytorch_correctness',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(Path(__file__).resolve().parent / 'oracle_generalization.py', controls / 'oracle_generalization.py')
    driver = output / 'driver.py'
    shutil.copy2(__file__, driver)
    apis = APIS if args.stage == 'final' else ('torch.sin', 'torch.mm')
    seconds = 180 if args.stage == 'final' else 90
    schedule = []
    for i, api in enumerate(apis):
        order = tuple(ARMS)[i % 4:] + tuple(ARMS)[:i % 4]
        for arm in order:
            schedule.append({'index': len(schedule), 'arm': arm, 'api': api, 'seed': 0,
                             'directory': f'{len(schedule):02d}-{arm}-{api.replace(".", "-")}'})
    protocol = {'version': 'exploration-oracle-experiment-1', 'stage': args.stage,
        'created_utc': utc(), 'engines': engines, 'controls_path': str(controls), 'controls_sha256': inventory(controls), 'driver': str(driver), 'driver_sha256': digest(driver),
        'arms': {k: {'representation_traversal': v[0], 'oracle_improvements': v[1]} for k, v in ARMS.items()},
        'schedule': schedule, 'seeds': [0], 'search_seconds': seconds, 'worker_timeout': 120,
        'cases': 100000, 'device': 'cpu', 'path_policy': 'backward', 'search_policy': 'adaptive',
        'generation_mode': 'local', 'coverage_basis': 'semantic', 'coverage_refinement': True,
        'replay_failures': 1, 'model': os.environ.get('MIRAGE_VLLM_MODEL'),
        'serving_config': str(args.serving_config.resolve()) if args.serving_config else None,
        'serving_config_sha256': digest(args.serving_config) if args.serving_config else None,
        'budget_scope': 'Search, worker execution and automatic replays share per-search budget; metadata separate.',
        'order': 'API blocks; arm order rotated by API index, fixed before final calls.',
        'limitations': ['One seed per API/arm; no significance or exhaustive-coverage claim.',
                       'No known defect triggers supplied; interrupted searches retained without rerun.',
                       'A uses the immutable pre-change engine; B/C/D use explicit feature flags.']}
    write(output / 'protocol.json', protocol)
    write(output / 'progress.json', {'state': 'locked', 'completed': [], 'expected': len(schedule)})
    print(str(output / 'protocol.json'))
    return 0


def execute_one(path, index):
    require_allocation()
    protocol = locked_protocol(path)
    item = protocol['schedule'][index]
    directory = path.parent / item['directory']
    directory.mkdir(exist_ok=False)
    write(directory / 'started.json', {'utc': utc(), 'job': os.environ['SLURM_JOB_ID'],
          'step': os.environ['SLURM_STEP_ID'], 'item': item, 'protocol_sha256': digest(path)})
    engine = protocol['engines']['baseline' if item['arm'] == 'A' else 'candidate']['path']
    sys.path.insert(0, engine)
    os.environ['PYTHONPATH'] = engine
    from mirage.workflow import run_stateful
    from mirage.contracts.programs import RunOptions
    from mirage.artifacts.report import summarize_search
    import torch
    serving_path = Path(os.environ.get('MIRAGE_SESSION_DIR', '.'))/'serving/environment.json'
    write(directory/'execution-environment.json', {'python': sys.version, 'executable': sys.executable,
          'torch': str(torch.__version__), 'torch_git': torch.version.git_version,
          'cuda_runtime': torch.version.cuda, 'hostname': os.uname().nodename,
          'model': os.environ.get('MIRAGE_VLLM_MODEL'), 'engine': engine,
          'serving_evidence': str(serving_path) if serving_path.exists() else None,
          'serving_evidence_sha256': digest(serving_path) if serving_path.exists() else None})
    started = time.monotonic()
    status = 2
    try:
        flags = {} if item['arm'] == 'A' else protocol['arms'][item['arm']]
        result = run_stateful(RunOptions(str(directory / 'search'), 'vllm',
            cases=protocol['cases'], time_limit=protocol['search_seconds'],
            worker_timeout=protocol['worker_timeout'], seed=item['seed'], device=protocol['device'],
            strategy='stateful', contract_version=4, replay_failures=protocol['replay_failures']),
            (item['api'],), path_policy=protocol['path_policy'],
            search_policy=protocol['search_policy'], generation_mode=protocol['generation_mode'],
            coverage_basis=protocol['coverage_basis'], coverage_refinement=protocol['coverage_refinement'], **flags)
        write(directory / 'native-metrics.json', summarize_search(result))
        status = 0 if result.get('records') and result['stop_reason'] in ('time_limit', 'case_limit') else 1
        return status
    finally:
        write(directory / 'completion.json', {'exit_status': status, 'wall_seconds': time.monotonic()-started,
              'finished_utc': utc(), 'job': os.environ['SLURM_JOB_ID'], 'step': os.environ['SLURM_STEP_ID']})


def remaining_seconds():
    end = os.environ.get('SLURM_JOB_END_TIME')
    if end and end.isdigit():
        return max(0, int(end) - time.time())
    value = subprocess.check_output(['squeue', '-h', '-j', os.environ['SLURM_JOB_ID'], '-o', '%L'], text=True).strip()
    days, clock = value.split('-', 1) if '-' in value else ('0', value)
    parts = [int(v) for v in clock.split(':')]
    if len(parts) == 2:
        parts.insert(0, 0)
    if len(parts) != 3:
        raise RuntimeError('Allocation remaining time unavailable')
    return int(days)*86400 + parts[0]*3600 + parts[1]*60 + parts[2]


def run(args):
    require_allocation()
    protocol = locked_protocol(args.protocol)
    completed, attempted, failed, launched = [], [], [], 0

    def checkpoint(state, **extra):
        write(args.protocol.parent/'progress.json', {'state': state, 'completed': completed,
              'attempted': attempted, 'failed_indices': failed, 'expected': len(protocol['schedule']),
              'updated_utc': utc(), **extra})

    def cancellation_unconfirmed():
        session = os.environ.get('MIRAGE_SESSION_DIR')
        return bool(session and (Path(session)/'cancellation-unconfirmed').exists())

    for item in protocol['schedule']:
        if cancellation_unconfirmed():
            checkpoint('cancellation_unconfirmed')
            return 2
        directory = args.protocol.parent/item['directory']
        if directory.exists():
            completion = directory/'completion.json'
            if not completion.exists():
                checkpoint('interrupted_without_completion', blocked_index=item['index'])
                raise RuntimeError(f'Interrupted search retained without rerun: {directory}')
            code = json.loads(completion.read_text())['exit_status']
            process_path = directory/'process.json'
            if process_path.exists():
                code = code or json.loads(process_path.read_text())['exit_status']
            attempted.append(item['index'])
            (failed if code else completed).append(item['index'])
            if code == 124 or (directory/'timeout.json').exists():
                checkpoint('timeout', blocked_index=item['index'])
                return 124
            continue
        reserve = protocol['search_seconds'] + protocol['worker_timeout'] + 90
        if launched >= args.max_runs or remaining_seconds() < reserve:
            checkpoint('awaiting_next_allocation_or_chunk')
            return 1 if failed else 0
        logs = args.protocol.parent/'driver-logs'
        logs.mkdir(exist_ok=True)
        with (logs/f'{item["index"]:02d}.stdout').open('x') as out, (logs/f'{item["index"]:02d}.stderr').open('x') as err:
            proc = subprocess.Popen([sys.executable, protocol['driver'], 'execute-one',
                  '--protocol', str(args.protocol), '--index', str(item['index'])], stdout=out, stderr=err,
                  start_new_session=True)
            try:
                proc.wait(timeout=protocol['search_seconds'] + protocol['worker_timeout'] + 30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                session = os.environ.get('MIRAGE_SESSION_DIR')
                if session:
                    (Path(session)/'cancellation-unconfirmed').write_text('Search process timed out; verify outstanding inference cancellation before service reuse.\n')
                write(directory/'timeout.json', {'exit_status': 124, 'utc': utc()})
                checkpoint('timeout', failed_index=item['index'])
                return 124
        launched += 1
        write(directory/'process.json', {'exit_status': proc.returncode, 'finished_utc': utc()})
        completion = directory/'completion.json'
        if not completion.exists():
            checkpoint('interrupted_without_completion', blocked_index=item['index'])
            return proc.returncode or 2
        code = json.loads(completion.read_text())['exit_status']
        attempted.append(item['index'])
        (failed if code or proc.returncode else completed).append(item['index'])
        checkpoint('running')
        if cancellation_unconfirmed():
            checkpoint('cancellation_unconfirmed')
            return 2
        if code == 124 or proc.returncode == 124:
            write(directory/'timeout.json', {'exit_status': 124, 'utc': utc()})
            checkpoint('timeout', failed_index=item['index'])
            return 124
        # Terminal ordinary failures stay recorded; independent later entries may run.
    return score(args.protocol)


def score(path):
    require_allocation()
    import torch
    protocol = locked_protocol(path)
    sys.path.insert(0, protocol['engines']['candidate']['path'])
    from mirage.artifacts.report import summarize_search
    rows = []
    for item in protocol['schedule']:
        directory = path.parent / item['directory']
        summary = directory / 'search' / 'summary.json'
        completion = directory / 'completion.json'
        common = None
        restored_followup_inspections = 0
        if summary.exists():
            common = json.loads(summary.read_text())
            for row in common.get('records', []):
                saved = directory/'search'/f"attempt-{row['attempt']:04d}"/'execution/followup.pt'
                if saved.exists() and 'followup_inspection' not in row:
                    record = torch.load(saved, weights_only=True, map_location='cpu')
                    if record.get('inspection'):
                        row['followup_inspection'] = record['inspection']
                        restored_followup_inspections += 1
        rows.append(dict(item, measured=summary.exists(), started=(directory/'started.json').exists(),
                         timed_out=(directory/'timeout.json').exists(),
                         completion=json.loads(completion.read_text()) if completion.exists() else None,
                         restored_saved_followup_inspections=restored_followup_inspections,
                         process_exit_status=json.loads((directory/'process.json').read_text())['exit_status']
                             if (directory/'process.json').exists() else 0,
                         metrics=summarize_search(common) if common is not None else None))
    all_attempted = all(r['started'] or r['completion'] is not None for r in rows)
    all_terminal = all(r['completion'] is not None for r in rows)
    incomplete = [r['index'] for r in rows if r['started'] and not r['completion']]
    all_successful = all(r['measured'] and r['completion'] and r['completion']['exit_status'] == 0
                         and r.get('process_exit_status', 0) == 0 for r in rows)
    failed = [r['index'] for r in rows if r['timed_out'] or
              (r['completion'] and (r['completion']['exit_status'] != 0 or r.get('process_exit_status', 0) != 0))]
    write(path.parent / 'comparison.json', {'version': 'common-score-1', 'complete': all_successful,
          'all_attempted': all_attempted, 'all_terminal': all_terminal, 'all_successful': all_successful,
          'failed_indices': failed, 'incomplete_indices': incomplete, 'runs': rows,
          'common_scorer_sha256': digest(Path(protocol['engines']['candidate']['path'])/'mirage/artifacts/report.py'),
          'aggregation': 'Counts remain per API and arm; distinct-unit counts must not be added as a global union. Legacy followup inspection restored only from original raw saved evidence, with no rerun or invented witnesses.'})
    write(path.parent / 'progress.json', {'state': 'complete' if all_successful else
          'all_attempted_with_failures' if all_terminal else 'incomplete',
          'all_attempted': all_attempted, 'all_terminal': all_terminal, 'all_successful': all_successful,
          'failed_indices': failed, 'incomplete_indices': incomplete,
          'attempted': [r['index'] for r in rows if r['started'] or r['completion']],
          'completed': [r['index'] for r in rows if r['completion'] and r['index'] not in failed],
          'expected': len(rows), 'updated_utc': utc()})
    return 0 if all_successful else 1


def reassess(args):
    require_allocation()
    protocol = locked_protocol(args.protocol)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    records = sorted({str(p.resolve()) for pattern in args.records for p in Path(args.protocol.parent).glob(pattern)})
    if not records:
        raise ValueError('No records matched; no measurement performed')
    write(output/'manifest.json', {'records': {p: digest(Path(p)) for p in records},
          'scope': 'Same saved record, old/new routing. Missing new witnesses stay unsupported. No new discovery.'})
    for name, engine in protocol['engines'].items():
        env = dict(os.environ, PYTHONPATH=engine['path'])
        command = """import json,sys,torch
from pathlib import Path
from mirage.inspector.oracles import audit_execution
manifest=json.loads(Path(sys.argv[1]).read_text())
results=[]
for path in manifest['records']:
 record=torch.load(path,weights_only=False,map_location='cpu')
 result=audit_execution(record,budget_seconds=10)
 results.append({'record':path,'audit':result})
Path(sys.argv[2]).write_text(json.dumps(results,indent=2)+'\\n')
"""
        subprocess.run([sys.executable, '-c', command, str(output/'manifest.json'), str(output/f'{name}.json')],
                       env=env, check=True)
    return 0


def controls(args):
    require_allocation()
    import importlib.util
    protocol = locked_protocol(args.protocol)
    engine = protocol['engines'][args.engine]['path']
    sys.path.insert(0, engine)
    os.environ['PYTHONPATH'] = engine
    root = Path(protocol['controls_path'])
    if args.kind == 'fixtures':
        sys.path.insert(0, str(root/'pytorch_correctness'))
        spec = importlib.util.spec_from_file_location('frozen_fixture_evaluator', root/'pytorch_correctness/evaluate.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        if args.engine == 'candidate':
            original_worker = module.run_worker
            def scoped_worker(request, *positional, **keywords):
                request = dict(request, config=dict(request.get('config', {}), oracle_improvements=True))
                return original_worker(request, *positional, **keywords)
            module.run_worker = scoped_worker
        code = module.stateful_integration(args.output, 600)
        independent = json.loads((args.output/'independent-controls.json').read_text())
        return int(code != 0 or not independent['all_ten_independently_corroborated'])
    if args.engine != 'candidate':
        raise ValueError('Generate new witnessed heldout records with candidate; reassess same records with baseline separately')
    spec = importlib.util.spec_from_file_location('frozen_heldout', root/'oracle_generalization.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    import torch
    from mirage.runtime.stateful import execute
    from mirage.runtime.probes import Recorder
    from mirage.inspector.oracles import audit_execution
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for family, api, call in module.HELDOUT:
        values = '[[[2.0,0.2],[0.1,1.5]]]' if api == 'torch.baddbmm' else '[[2.0,0.2],[0.1,1.5]]'
        source = f"""import torch

def build(device, seed):
    x = torch.tensor({values}, dtype=torch.float64, device=device, requires_grad=True)
    y = torch.tensor({values}, dtype=torch.float64, device=device, requires_grad=True)
    optimizer = torch.optim.SGD([x, y], lr=0.1, momentum=0.0, weight_decay=0.0, dampening=0.0, nesterov=False, maximize=False, foreach=False, fused=False, differentiable=False)
    return {{'inputs': {{'x': x, 'y': y}}, 'parameters': {{'x': x, 'y': y}}, 'optimizer': optimizer}}

def run(ctx, probe):
    x = ctx['inputs']['x']
    y = ctx['inputs']['y']
    out = probe.call({call})
    out.sum().backward()
    ctx['optimizer'].step()
    return {{'output': out, 'gradient': x.grad}}
"""
        folder = args.output/api
        folder.mkdir()
        (folder/'program.py').write_text(source)
        config = dict(api=api, device='cpu', seed=0, requires_backward=True, oracle_improvements=True)
        record = execute(source, config, Recorder(folder/'trace.jsonl'))
        audit = audit_execution(record, budget_seconds=30)
        torch.save(record, folder/'record.pt')
        write(folder/'audit.json', audit)
        rows.append({'api': api, 'family': family, 'valid': record['inspection']['valid'], 'audit': audit})
        print(api, audit['status'], audit.get('reason'), flush=True)
    passed = all(r['valid'] and r['audit']['status'] == 'agreement' for r in rows)
    write(args.output/'summary.json', {'records': rows, 'passed': passed,
          'scope': 'Eight heldout APIs, native SGD after backward; sampled consistency controls, not autonomous discoveries.'})
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('lock')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--stage', choices=('development', 'final'), required=True)
    p.add_argument('--serving-config', type=Path)
    for name in ('run', 'execute-one', 'score', 'reassess', 'controls'):
        p = sub.add_parser(name)
        p.add_argument('--protocol', type=Path, required=True)
        if name == 'run': p.add_argument('--max-runs', type=int, default=32)
        if name == 'execute-one': p.add_argument('--index', type=int, required=True)
        if name == 'controls':
            p.add_argument('--kind', choices=('heldout', 'fixtures'), required=True)
            p.add_argument('--engine', choices=('baseline', 'candidate'), default='candidate')
            p.add_argument('--output', type=Path, required=True)
        if name == 'reassess':
            p.add_argument('--output', type=Path, required=True)
            p.add_argument('--records', nargs='+', required=True)
    args = parser.parse_args()
    if hasattr(args, 'protocol'): args.protocol = args.protocol.resolve()
    if args.command == 'lock': return lock(args)
    if args.command == 'run': return run(args)
    if args.command == 'execute-one': return execute_one(args.protocol, args.index)
    if args.command == 'score': return score(args.protocol)
    if args.command == 'controls': return controls(args)
    return reassess(args)


if __name__ == '__main__':
    raise SystemExit(main())

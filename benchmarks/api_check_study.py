"""Paired blind API-check evaluation. This driver is outside production generation."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import re
import shutil
import signal
import subprocess
import sys
import time

from e2e_latency import confirmed_idle, digest, inventory, read, require_allocation, targets, write

ARMS = ('stateful', 'check_primary', 'check_fallback')
SEEDS = (13, 37)


def holdouts(code, documentation, known):
    import torch
    # Exclude even incidental mentions of a basename in development material.
    seen = set()
    for directory in (code/'src', code/'tests', code/'benchmarks', documentation):
        for path in directory.rglob('*'):
            if path.suffix in ('.md', '.py', '.json') and path.is_file():
                seen.update(re.findall(r'\b[A-Za-z_][A-Za-z_0-9]*\b', path.read_text(errors='replace')))
    seen.update(item['api'].split('.')[-1] for item in known)
    groups = {}
    rng = random.Random(20261118)
    for namespace, module in [('torch.special', torch.special), ('torch.fft', torch.fft), ('Tensor', torch.Tensor)]:
        candidates = []
        for name in sorted(dir(module)):
            if name.startswith('_') or name.endswith('_') or name in seen:
                continue
            value = getattr(module, name)
            if not callable(value) or not value.__doc__:
                continue
            if namespace == 'Tensor':
                packet = getattr(torch.ops.aten, name, None)
                overload = getattr(packet, 'default', None)
                if overload is None or torch.Tag.pointwise not in overload.tags:
                    continue
                schema = overload._schema
                if len(schema.arguments) != 1 or str(schema.arguments[0].type) != 'Tensor':
                    continue
            elif getattr(value, '__module__', None) not in (namespace, 'torch._C._special', 'torch._C._fft'):
                continue
            candidates.append(namespace + '.' + name)
        if not candidates:
            raise ValueError('No unseen candidate in stratum ' + namespace)
        groups[namespace] = dict(pool=candidates, selected=rng.choice(candidates))
    return groups


def prepare(root, baseline, docs):
    from mirage.inspector.comparison import DEVICE_POLICY
    from mirage.runtime.environment import describe
    code = Path(__file__).resolve().parents[1]
    published = read(baseline.parents[1]/'source-sha256.json')
    expected = {name[4:]:sha for name, sha in published.items() if name.startswith('src/') and name.endswith('.py')}
    if not expected or inventory(baseline) != expected:
        raise ValueError('Stateful baseline differs from its frozen source manifest')
    root.mkdir(parents=True, exist_ok=False)
    for name, origin in [('engine', code/'src'), ('baseline', baseline), ('benchmarks', code/'benchmarks')]:
        shutil.copytree(origin, root/name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    known = targets(read(code/'benchmarks/pytorch_correctness/candidates.json'))
    strata = holdouts(code, docs, known)
    roster = [dict(row, group='benchmark') for row in known]
    roster.extend(dict(id='heldout-' + group, api=item['selected'], device='cpu', group='heldout')
                  for group, item in strata.items())
    tasks = []
    for seed_i, seed in enumerate(SEEDS):
        shuffled = list(roster)
        random.Random(seed).shuffle(shuffled)
        for index, target in enumerate(shuffled):
            rotation = (index + seed_i) % len(ARMS)
            for arm in ARMS[rotation:] + ARMS[:rotation]:
                tasks.append(dict(target, arm=arm, seed=seed, index=len(tasks)))
    session = Path(os.environ['MIRAGE_SESSION_DIR'])
    protocol = dict(version='paired-fragments-1', tasks=tasks, seeds=SEEDS, roster=roster,
        strata=strata, heldout_sampling='sorted installed public callables, seeded uniform one per stratum after excluding development basenames',
        arm_order='paired seeds, shuffled API blocks, rotating arm order, shared warm service',
        cases=4, seconds=120, worker_timeout=75, generation_concurrency=1,
        generation_policy='stateful uses the whole wall deadline; both check arms use the same 20 percent differential reserve (45s cap) when CUDA is available',
        controlled_differences='full legacy stateful vs Spec/fragment changes is a compound comparison; paired check arms isolate enabling fallback',
        code_root=str(root),
        sampling='temperature 0, top_p 1; paired seeds govern exploration, completions may coincide',
        engine_hashes={k:inventory(root/k) for k in ('engine','baseline','benchmarks')},
        model=read(session/'serving/environment.json'), execution=describe('cuda'),
        job=os.environ['SLURM_JOB_ID'], step=os.environ['SLURM_STEP_ID'],
        policy=DEVICE_POLICY, max_tokens=int(os.environ['MIRAGE_MAX_TOKENS']),
        fixed_controls=read(session/'fixed-controls/summary.json')['cases'],
        stop='4 attempts or 120 seconds from cell start, worker cap 75s, 30s process cleanup allowance; keep every cell and failure',
        scope='Budgeted pilot. A reproducible benchmark detection requires the existing independent mechanism matcher and verified replay. No evidence of a novel defect from agreement or mismatch alone.')
    write(root/'protocol.json', protocol)
    (root/'protocol.sha256').write_text(digest(root/'protocol.json'))
    return protocol


def baseline_cell(api, device, seed, directory, deadline):
    from mirage.contracts.programs import RunOptions
    from mirage.workflow import run_stateful
    options = RunOptions(str(directory), 'vllm', cases=4, time_limit=120, worker_timeout=75,
                         device=device, seed=seed, strategy='stateful', contract_version=4)
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    report = run_stateful(options, (api,), path_policy='mixed', representation_traversal=True,
                          oracle_improvements=True, wall_deadline=deadline)
    failed = any(r.get('status') == 'provider_failed' for r in report['records'])
    return int(failed or report['stop_reason'] in ('metadata_failed','controller_error','provider_cancellation_unconfirmed'))


def measure(root, protocol, task):
    cell = root/'cells'/('%03d' % task['index'])
    cell.mkdir(parents=True, exist_ok=False)
    arm = task['arm']
    command = ([sys.executable, str(root/'benchmarks/api_check_study.py'), 'baseline-cell',
                str(root), '--index', str(task['index'])] if arm == 'stateful' else
               [sys.executable, '-m', 'mirage', 'check', task['api']])
    env = dict(os.environ, PYTHONPATH=str(root/('baseline' if arm=='stateful' else 'engine')),
        MIRAGE_PROVIDER='vllm', MIRAGE_CHECK_TIME_LIMIT=str(protocol['seconds']),
        MIRAGE_CHECK_CASES=str(protocol['cases']), MIRAGE_CHECK_WORKER_TIMEOUT=str(protocol['worker_timeout']),
        MIRAGE_CHECK_DEVICE=task['device'], MIRAGE_CHECK_SEED=str(task['seed']),
        MIRAGE_CHECK_DIFFERENTIAL=str(int(arm=='check_fallback')), MIRAGE_CHECK_OUTPUT_ROOT=str(cell/'check'))
    start = time.monotonic()
    env['MIRAGE_STUDY_DEADLINE'] = str(start + protocol['seconds'])
    write(cell/'dispatch.json', dict(task, argv=command, monotonic_start=start))
    with (cell/'stdout.log').open('w') as out, (cell/'stderr.log').open('w') as err:
        with subprocess.Popen(command, env=env, stdout=out, stderr=err) as child:
            try:
                code = child.wait(timeout=protocol['seconds'] + 20)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                code = 124
    write(cell/'measurement.json', dict(task, exit_status=code, e2e_seconds=time.monotonic()-start))
    return cell


def score(cell):
    from mirage.artifacts.bundle import load_tensor_record
    from mirage.artifacts.replay import primary_route
    from pytorch_correctness.evaluate import stateful_independent_audit, stateful_attribution
    row = read(cell/'measurement.json')
    check_path = next((cell/'check').glob('*/summary.json'), None)
    check = read(check_path) if check_path else {}
    native_path = (check_path.parent/'execution/summary.json' if check_path else cell/'native/summary.json')
    native = read(native_path) if native_path.is_file() else {}
    cases = native.get('records', [])
    search = native.get('search_metrics', {})
    reported, replayed, attributable = [], [], []
    seen_mechanism = False
    unknown_mechanism = False
    independent_positive = False
    for case in cases:
        valid = case.get('inspection', {}).get('valid')
        suspected = case.get('verdict') == 'suspected_failure'
        stable = any(r.get('replay_valid') is True for r in case.get('automatic_replays', []))
        if suspected:
            reported.append(case['attempt'])
        if suspected and stable:
            replayed.append(case['attempt'])
        if row['group'] != 'benchmark' or not valid:
            continue
        for member in ('source', 'followup'):
            path = Path(case['directory'])/'execution'/(member+'.pt')
            if path.is_file():
                record = load_tensor_record(path)
                audit = stateful_independent_audit(row['id'], record)
                result = stateful_attribution(row['id'], record, audit)
                seen_mechanism |= result['structural_trigger'] is True
                unknown_mechanism |= result['structural_trigger'] is None
                independent_positive |= result['benchmark_reproduction']
                if suspected and stable and result['benchmark_reproduction']:
                    attributable.append(dict(attempt=case['attempt'], member=member))
    diff = check.get('differential', {}).get('rows', [])
    counts = Counter(item['status'] for item in diff)
    for item in diff:
        if item['status'] != 'stable_mismatch':
            continue
        reported.append(item['attempt'])
        replayed.append(item['attempt'])
        if row['group'] == 'benchmark':
            for device in ('cpu', 'cuda'):
                path = Path(item['directory'])/('0-' + device)/'member.pt'
                record = load_tensor_record(path)
                audit = stateful_independent_audit(row['id'], record)
                if stateful_attribution(row['id'], record, audit)['benchmark_reproduction']:
                    attributable.append(dict(attempt=item['attempt'], member=device))
                    independent_positive = seen_mechanism = True
    primary = sum(primary_route(case) in ('primary_mr','primary_numerical') for case in cases)
    if row['arm'] == 'stateful':
        primary = sum(bool(case.get('inspection', {}).get('valid')) and
                      (case.get('mr_verdict') in ('pass','suspected_failure') or
                       any(a.get('status') in ('agreement','violation') for a in case.get('gradient_audits',{}).values()))
                      for case in cases)
    provider_timings = [read(path) for path in native_path.parent.glob('provider-*/provider-timing.json')]
    metrics = dict(row, terminal=bool(check.get('terminal') if check else native.get('stop_reason')),
        status=check.get('status', native.get('stop_reason','incomplete')),
        reason=check.get('reason', native.get('stop_reason')),
        attempts=len(cases), reached=search.get('runtime_valid',0), primary=primary,
        differential=dict(counts), judgments=primary+counts['agreement']+counts['stable_mismatch'],
        mr=search.get('applicable_comparisons',0), numerical=search.get('numerical_audit_statuses',{}),
        reported=reported,replayed=replayed, benchmark_matches=attributable,
        benchmark_mechanism_observed=(None if unknown_mechanism and not seen_mechanism else seen_mechanism)
            if row['group']=='benchmark' else None,
        benchmark_positive=independent_positive if row['group']=='benchmark' else None,
        first_missing_stage=('execution' if not search.get('runtime_valid') else 'mechanism_unmeasured'
            if unknown_mechanism and not seen_mechanism else 'mechanism'
            if not seen_mechanism else 'oracle' if not independent_positive else 'replay_attribution'
            if not attributable else None) if row['group']=='benchmark' else None,
        reached_relations=search.get('relational_goal_attained_cases'), blocking=search.get('blocking_stages',{}),
        tokens=search.get('known_tokens'), tokens_complete=search.get('token_usage_complete'),
        model_seconds=sum(t['elapsed_seconds'] for t in provider_timings), model_calls=native.get('provider_calls',0),
        worker_seconds=sum(r.get('process',{}).get('elapsed_seconds',0.) for r in cases),
        manual_edits=0, run_directory=str(native_path.parent))
    write(cell/'score.json', metrics)
    return metrics


def campaign(root):
    if Path(__file__).resolve() != (root/'benchmarks/api_check_study.py').resolve():
        raise ValueError('Launch the frozen campaign driver')
    sys.path.insert(0, str(root/'engine'))
    protocol = read(root/'protocol.json')
    if digest(root/'protocol.json') != (root/'protocol.sha256').read_text():
        raise ValueError('Protocol checksum changed')
    if any(inventory(root/k) != hashes for k, hashes in protocol['engine_hashes'].items()):
        raise ValueError('Frozen code changed')
    rows = []
    for task in protocol['tasks']:
        cell = root/'cells'/('%03d' % task['index'])
        if cell.exists() and not (cell/'measurement.json').exists():
            # Preserve an interrupted dispatch in the denominator; no duplicate model call.
            write(cell/'measurement.json', dict(task, exit_status=130, e2e_seconds=None))
        if not cell.exists():
            until = time.monotonic()+60
            while not confirmed_idle():
                if time.monotonic() >= until:
                    raise RuntimeError('No confirmed idle service boundary')
                time.sleep(1)
            cell = measure(root, protocol, task)
        rows.append(read(cell/'score.json') if (cell/'score.json').exists() else score(cell))
        write(root/'progress.json', dict(expected=len(protocol['tasks']), scored=len(rows), last=task))
        print(f"{len(rows)}/{len(protocol['tasks'])} {task['arm']} {task['api']} seed={task['seed']} valid={rows[-1]['reached']}", flush=True)
    grouped = []
    for group in ('benchmark','heldout'):
        for arm in ARMS:
            cells = [r for r in rows if r['arm']==arm and r['group']==group]
            elapsed = sum(r['e2e_seconds'] or 0 for r in cells)
            attempts = sum(r['attempts'] for r in cells)
            grouped.append(dict(group=group, arm=arm, cells=len(cells),
                unattended_terminal=sum(r['terminal'] for r in cells),
                successful_cells=sum(r['terminal'] and r['exit_status']==0 for r in cells),
                executed_cells=sum(bool(r['reached']) for r in cells),
                executed_apis=sorted({r['api'] for r in cells if r['reached']}),
                attempts=attempts, valid=sum(r['reached'] for r in cells),
                e2e_by_cell_seconds=[r['e2e_seconds'] for r in cells],
                reach_rate=sum(r['reached'] for r in cells)/attempts if attempts else None,
                judged_per_reached=sum(r['judgments'] for r in cells)/sum(r['reached'] for r in cells)
                    if any(r['reached'] for r in cells) else None,
                failure_stages=dict(Counter(r['first_missing_stage'] for r in cells if r['first_missing_stage'])),
                primary=sum(r['primary'] for r in cells), judgments=sum(r['judgments'] for r in cells),
                differential=dict(sum((Counter(r['differential']) for r in cells),Counter())),
                reported_apis=sorted({r['api'] for r in cells if r['reported']}),
                replayed_apis=sorted({r['api'] for r in cells if r['replayed']}),
                benchmark_reproduced=sorted({r['id'] for r in cells if r['benchmark_matches']}),
                e2e_seconds=elapsed, judgments_per_hour=sum(r['judgments'] for r in cells)*3600/elapsed if elapsed else None,
                case_specific_edits=0))
    write(root/'results.json', dict(protocol_sha256=digest(root/'protocol.json'), arms=grouped, cells=rows))


def main():
    require_allocation()
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare','campaign','baseline-cell'))
    parser.add_argument('root',type=Path)
    parser.add_argument('--baseline',type=Path)
    parser.add_argument('--docs',type=Path)
    parser.add_argument('--index',type=int)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.root, args.baseline, args.docs)
    elif args.mode == 'campaign':
        campaign(args.root)
    else:
        task = read(args.root/'protocol.json')['tasks'][args.index]
        return baseline_cell(task['api'], task['device'], task['seed'],
            args.root/'cells'/('%03d'%args.index)/'native', float(os.environ['MIRAGE_STUDY_DEADLINE']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

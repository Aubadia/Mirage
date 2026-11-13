"""Bounded fresh-process reproduction; run only inside an allocated Slurm step."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


HERE = Path(__file__).resolve().parent


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def serialize(value):
    import torch
    if isinstance(value, torch.Tensor):
        t = value.detach().cpu().resolve_conj().resolve_neg()
        flat = t.flatten()
        return dict(shape=list(t.shape), dtype=str(t.dtype),
                    values=serialize(flat[:64].tolist()), truncated=flat.numel() > 64)
    if isinstance(value, complex):
        return dict(real=value.real, imag=value.imag)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(k): serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize(v) for v in value]
    return value


def environment():
    import torch
    return dict(torch=str(torch.__version__), torch_git=torch.version.git_version,
                python=sys.version, executable=sys.executable, cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), mkl=torch.backends.mkl.is_available(),
                mkldnn=torch.backends.mkldnn.is_available(), hostname=os.uname().nodename,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                threads=torch.get_num_threads(), slurm_job=os.environ.get('SLURM_JOB_ID'),
                slurm_step=os.environ.get('SLURM_STEP_ID'))


def worker(case, variant, directory):
    from cases import CASES
    import torch
    started = time.monotonic()
    record = dict(case=case, variant=variant, pid=os.getpid(),
                  started_utc=datetime.now(timezone.utc).isoformat(), environment=environment())
    try:
        value = CASES[case](variant)
        torch.save(value, directory / 'evidence.pt')
        record.update(serialize(value))
    except Exception as error:
        record.update(outcome='execution_error', error=f'{type(error).__name__}: {error}')
        traceback.print_exc()
    record['elapsed_seconds'] = time.monotonic()-started
    write(directory / 'result.json', record)
    print(json.dumps(record))
    return 3 if record['outcome'] == 'violation' else (0 if record['outcome'] == 'no_violation' else 2)


def kill_group(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def execute(command, directory, timeout):
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    process = None
    try:
        with (directory / 'stdout.txt').open('wb') as out, (directory / 'stderr.txt').open('wb') as err:
            process = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True)
            try:
                code = process.wait(timeout=timeout)
                status = 'exited'
            except subprocess.TimeoutExpired:
                kill_group(process)
                code = process.wait()
                status = 'timeout'
    finally:
        if process is not None:
            kill_group(process)
            process.wait()
    result = dict(status=status, returncode=code, elapsed_seconds=time.monotonic()-started, command=command)
    write(directory / 'process.json', result)
    return result


def classify(records, repetitions, variants):
    """Missing, duplicated, or conflicting cells cannot qualify as reproduced."""
    expected = {(r, v) for r in range(repetitions) for v in variants}
    cells = [(r['repeat'], r['variant']) for r in records]
    if len(cells) != len(expected) or set(cells) != expected:
        return 'incomplete'
    outcomes = {r['outcome'] for r in records}
    if outcomes == {'violation'}:
        return 'reproduced'
    if outcomes == {'no_violation'}:
        return 'not_reproduced'
    if outcomes <= {'violation', 'no_violation'}:
        return 'unstable_or_variant_dependent'
    return 'unresolved'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--repetitions', type=int, default=1)
    parser.add_argument('--variants', nargs='+', type=int, default=[0])
    parser.add_argument('--timeout', type=float, default=240.)
    parser.add_argument('--time-limit', type=float, default=12000.)
    parser.add_argument('--worker')
    parser.add_argument('--variant', type=int, default=0)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_STEP_ID'):
        parser.error('Execute computational work with srun inside Slurm.')
    if args.worker:
        return worker(args.worker, args.variant, args.output)
    if args.repetitions < 1 or not math.isfinite(args.timeout) or args.timeout <= 0 or not math.isfinite(args.time_limit) or args.time_limit <= 0 or any(v not in (0, 1) for v in args.variants) or len(set(args.variants)) != len(args.variants):
        parser.error('Positive budgets/repetitions and unique variants 0/1 are required.')
    catalog = json.loads((HERE / 'candidates.json').read_text())
    selected = args.cases or catalog.get('selected') or [c['id'] for c in catalog['candidates'] if c.get('implemented')]
    if not selected or len(set(selected)) != len(selected) or any(c not in {x['id'] for x in catalog['candidates'] if x.get('implemented')} for c in selected):
        parser.error('Unknown, empty or duplicate case selection.')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write(args.output / 'configuration.json', {**vars(args), 'output': str(args.output)})
    write(args.output / 'environment.json', environment())
    records = []
    def save():
        summary = dict(elapsed_seconds=time.monotonic()-started,
                       cases={c: classify([r for r in records if r['case'] == c], args.repetitions, args.variants) for c in selected},
                       records=records, confirmed_framework_defect=False,
                       note='Reproduced means the fixed-input case violates with passing controls and trigger predicates; review of independent oracles and root grouping remains required. Default: one execution, no randomness.')
        write(args.output / 'summary.json', summary)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        for case in selected:
            for repeat in range(args.repetitions):
                for variant in args.variants:
                    remaining = args.time_limit-(time.monotonic()-started)
                    if remaining < 5:
                        return 2
                    directory = args.output / case / f'repeat-{repeat}-variant-{variant}'
                    process = execute([sys.executable, str(HERE / 'run.py'), '--worker', case,
                                       '--variant', str(variant), '--output', str(directory)],
                                      directory, min(args.timeout, remaining))
                    record = dict(case=case, repeat=repeat, variant=variant, directory=str(directory),
                                  outcome='execution_error')
                    path = directory / 'result.json'
                    if process['status'] == 'timeout':
                        record['outcome'] = 'timeout'
                    elif path.exists():
                        value = json.loads(path.read_text())
                        code = 3 if value.get('outcome') == 'violation' else (0 if value.get('outcome') == 'no_violation' else 2)
                        if process['returncode'] == code:
                            record['outcome'] = value['outcome']
                    records.append(record)
                    print(case, repeat, variant, record['outcome'], flush=True)
                    save()
    finally:
        save()
    if any(r['outcome'] not in ('violation', 'no_violation') for r in records):
        return 2
    return 3 if any(r['outcome'] == 'violation' for r in records) else 0


if __name__ == '__main__':
    raise SystemExit(main())

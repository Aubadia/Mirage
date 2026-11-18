"""Frozen ten-API live E2E measurement; issue fixtures never enter generation."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(root):
    return {str(p.relative_to(root)): digest(p) for p in sorted(Path(root).rglob('*.py'))
            if '__pycache__' not in p.parts}


def targets(catalog):
    selected = catalog['selected']
    by_id = {c['id']: c for c in catalog['candidates']}
    if len(selected) != 10 or len(set(selected)) != 10:
        raise ValueError('Expected ten distinct selected benchmark IDs')
    # Do not copy issue conditions, examples, titles or oracle hints into prompts.
    return [{'id': key, 'api': by_id[key]['api'], 'device': by_id[key]['device']}
            for key in selected]


def prepare(args):
    if args.budget <= 0 or min(args.workers, getattr(args, 'active_apis', 1), getattr(args, 'generation_concurrency', 2)) < 1:
        raise ValueError('Positive model budget and worker limit required')
    args.output.mkdir(parents=True, exist_ok=False)
    engine = args.output / 'engine'
    shutil.copytree(args.engine.resolve(), engine,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    driver = args.output / 'driver.py'
    shutil.copy2(__file__, driver)
    roster = targets(read(Path(__file__).parent/'pytorch_correctness/candidates.json'))
    protocol = {'version': 'e2e-latency-2', 'shared_scheduling': True, 'label': args.label, 'targets': roster,
        'engine': str(engine.resolve()), 'engine_sha256': inventory(engine),
        'driver': str(driver.resolve()), 'driver_sha256': digest(driver),
        'model_budget_seconds': args.budget, 'workers': args.workers,
        'worker_timeout': 120, 'cases': 100000, 'seed': 0, 'replay_failures': 1,
        'path_policy': 'mixed', 'search_policy': 'adaptive', 'generation_mode': 'local',
        'coverage_basis': 'semantic', 'coverage_refinement': True,
        'representation_traversal': True, 'oracle_improvements': True,
        'generation_pipeline': True, 'generation_overlap': getattr(args, 'generation_overlap', True),
        'model': 'Qwen/Qwen3.8-27B',
        'model_revision': '37561816e66b5b9153b1b64c10968c18c1e6f4a0',
        'max_tokens': 4096, 'max_model_len': 16384, 'mtp_tokens': 3,
        'temperature': 0.0, 'top_p': 1.0, 'api_concurrency': getattr(args, 'active_apis', 1),
        'generation_concurrency': getattr(args, 'generation_concurrency', 2),
        'created_epoch': time.time(),
        'budget_scope': 'Cumulative provider request wall seconds per API; framework/backpressure separate.',
        'e2e_scope': 'Per-API child process launch to exit, including metadata, generation, all representation drain and durable report; scoring/validation and service startup separate.',
        'quality_scope': 'Live equal-budget quality and fixed-work equivalence are separate; no fixture hints, no exhaustive coverage or statistical-equivalence claim.'}
    write(args.output/'protocol.json', protocol)
    print(str(args.output/'protocol.json'), flush=True)


def verify(path):
    protocol = read(path)
    if inventory(Path(protocol['engine'])) != protocol['engine_sha256']:
        raise ValueError('Frozen engine changed')
    if digest(protocol['driver']) != protocol['driver_sha256']:
        raise ValueError('Frozen driver changed')
    return protocol


def require_allocation():
    if not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Computational phases require a Slurm step')


def metric_values(text):
    wanted = ('num_requests_', 'kv_cache_usage', 'prefix_cache_', 'request_',
              'time_to_first_token', 'inter_token_latency', 'e2e_request_latency',
              'generation_tokens', 'prompt_tokens', 'spec_decode', 'num_preemptions')
    values = {}
    for line in text.splitlines():
        if not line.startswith('vllm:') or not any(x in line.split('{', 1)[0] for x in wanted):
            continue
        match = re.match(r'^(\S+(?:\{.*\})?)\s+([-+.0-9eE]+)(?:\s|$)', line)
        if match:
            values[match[1]] = float(match[2])
    return values


def service_metrics():
    base = os.environ['MIRAGE_VLLM_BASE_URL'].removesuffix('/v1')
    with urllib.request.urlopen(base+'/metrics', timeout=5) as response:
        return response.read().decode()


def idle():
    values = metric_values(service_metrics())
    running = [v for k,v in values.items() if k.startswith('vllm:num_requests_running{')]
    waiting = [v for k,v in values.items() if k.startswith('vllm:num_requests_waiting{')]
    return bool(running and waiting) and all(v == 0 for v in running + waiting)


def confirmed_idle():
    return idle() and idle()


def model_environment(protocol):
    serving = read(Path(os.environ['MIRAGE_SESSION_DIR'])/'serving/environment.json')
    for key in ('model', 'mtp_tokens', 'max_model_len'):
        if serving[key] != protocol[key]:
            raise ValueError('Serving differs from protocol: '+key)
    if serving['declared_revision'] != protocol['model_revision']:
        raise ValueError('Model revision differs')
    if int(os.environ.get('MIRAGE_MAX_TOKENS', '4096')) != protocol['max_tokens']:
        raise ValueError('Output allowance differs')
    for key, default, expected in (('MIRAGE_TEMPERATURE','0',protocol['temperature']),
                                   ('MIRAGE_TOP_P','1',protocol['top_p'])):
        if float(os.environ.get(key, default)) != expected:
            raise ValueError('Sampling differs: '+key)
    return serving


def one(args):
    require_allocation()
    protocol = verify(args.protocol)
    sys.path.insert(0, protocol['engine'])
    from mirage.contracts.programs import RunOptions
    from mirage.workflow import run_stateful
    target = protocol['targets'][args.index]
    directory = args.protocol.parent/f"api-{args.index:02d}-{target['api'].replace('.', '-')}"
    directory.mkdir(exist_ok=False)
    environment = model_environment(protocol)
    write(directory/'started.json', {'epoch': time.time(), 'target': target,
        'job': os.environ['SLURM_JOB_ID'], 'step': os.environ['SLURM_STEP_ID'],
        'serving': environment, 'protocol_sha256': digest(args.protocol)})
    options = RunOptions(str(directory/'search'), 'vllm', cases=protocol['cases'],
        time_limit=protocol['model_budget_seconds'], worker_timeout=protocol['worker_timeout'],
        seed=protocol['seed'], device=target['device'], strategy='stateful', contract_version=4,
        workers=protocol['workers'], replay_failures=protocol['replay_failures'], generation_pipeline=True,
        generation_overlap=protocol.get('generation_overlap', True))
    started = time.monotonic()
    status = 2
    try:
        result = run_stateful(options, (target['api'],), **{k:protocol[k] for k in
            ('path_policy','search_policy','generation_mode','coverage_basis','coverage_refinement',
             'representation_traversal','oracle_improvements')})
        run_seconds = time.monotonic()-started
        write(directory/'metrics.json', {'api': target['api'], 'run_seconds': run_seconds,
            'search_metrics': result['search_metrics'], 'pipeline_timing': result.get('pipeline_timing'),
            'stop_reason': result.get('stop_reason'), 'pipeline_complete': result.get('pipeline_complete')})
        status = 0 if result.get('pipeline_complete') and result.get('stop_reason') in ('llm_time_limit','case_limit') else 1
        return status
    finally:
        write(directory/'completion.json', {'exit_status': status, 'finished_epoch': time.time(),
            'child_seconds': time.monotonic()-started})


def remaining():
    value = subprocess.check_output(['squeue','-h','-j',os.environ['SLURM_JOB_ID'],'-o','%L'], text=True).strip()
    days, clock = value.split('-', 1) if '-' in value else ('0', value)
    parts = [int(n) for n in clock.split(':')]
    if len(parts) == 2: parts.insert(0, 0)
    if len(parts) != 3: raise ValueError('Unrecognized allocation time')
    return int(days)*86400+parts[0]*3600+parts[1]*60+parts[2]


def parallel_campaign(protocol, root):
    sys.path.insert(0, protocol['engine'])
    from mirage.contracts.programs import RunOptions
    from mirage.contracts.scheduling import APIJob, SchedulingOptions
    from mirage.runtime.multi_api import run_sessions
    jobs = []
    for index, target in enumerate(protocol['targets']):
        directory = root / f"api-{index:02d}-{target['api'].replace('.', '-')}"
        options = RunOptions(str(directory / 'search'), 'vllm', strategy='stateful', contract_version=4,
            device=target['device'], time_limit=protocol['model_budget_seconds'],
            workers=protocol['workers'], generation_concurrency=protocol.get('generation_concurrency', 2),
            worker_timeout=protocol['worker_timeout'], cases=protocol['cases'], seed=protocol['seed'],
            replay_failures=protocol['replay_failures'], generation_pipeline=True,
            generation_overlap=protocol.get('generation_overlap', True))
        treatment = {key: protocol[key] for key in ('path_policy', 'search_policy', 'generation_mode',
            'coverage_basis', 'coverage_refinement', 'representation_traversal', 'oracle_improvements')}
        jobs.append(APIJob(f'api-{index:03d}', target['api'], options, treatment))
    limits = SchedulingOptions(protocol['workers'], protocol.get('generation_concurrency', 2), protocol['api_concurrency'])
    result = run_sessions(jobs, limits, root / 'parallel', deadline=time.monotonic() + max(0, remaining() - 300))
    for row in result['sessions']:
        directory = Path(row['directory']).parent
        write(directory / 'metrics.json', {'session': {k: v for k, v in row.items() if k != 'report'},
                                           'report': row.get('report')})
    return result


def campaign(args):
    require_allocation()
    protocol = verify(args.protocol)
    model_environment(protocol)
    root = args.protocol.parent
    if (root/'campaign.json').exists():
        raise ValueError('Campaign already attempted; preserve it without an automatic rerun')
    stop = threading.Event()
    cancelled = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_:cancelled.set())
    def sample():
        with (root/'service-samples.jsonl').open('x') as stream:
            while not stop.is_set():
                try: data = {'epoch':time.time(), 'metrics':metric_values(service_metrics())}
                except (OSError,ValueError) as error: data = {'epoch':time.time(), 'error':str(error)}
                stream.write(json.dumps(data)+'\n');stream.flush()
                stop.wait(2)
    thread = threading.Thread(target=sample, daemon=True);thread.start()
    started = time.monotonic(); rows=[]; state='running'
    write(root/'campaign.json', {'state':state,'completed':rows,'expected':len(protocol['targets'])})
    try:
        if protocol.get('shared_scheduling') or protocol.get('api_concurrency', 1) > 1:
            if not confirmed_idle():
                state='service_not_idle'
                return 1
            result=parallel_campaign(protocol, root)
            rows.extend(result['sessions'])
            service_idle=confirmed_idle()
            state='complete' if result['stop_reason']=='completed' and service_idle else 'failed_or_incomplete'
            verify(args.protocol)
            return 0 if state=='complete' else 1
        for i,target in enumerate(protocol['targets']):
            if cancelled.is_set() or remaining() < 1200:
                state='interrupted';break
            if not confirmed_idle():
                state='service_not_idle';break
            label = f"api-{i:02d}-{target['api'].replace('.', '-')}"
            (root/(label+'-metrics-before.txt')).write_text(service_metrics())
            begin=time.monotonic();begin_epoch=time.time()
            env=dict(os.environ, PYTHONPATH=protocol['engine'], OMP_NUM_THREADS='1',
                     MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                     TORCHINDUCTOR_COMPILE_THREADS='1')
            with (root/(label+'.stdout.log')).open('wb') as out, (root/(label+'.stderr.log')).open('wb') as err:
                process=subprocess.Popen([sys.executable,protocol['driver'],'one',
                    '--protocol',str(args.protocol.resolve()),'--index',str(i)],stdout=out,stderr=err,env=env)
                terminated=False;term_at=None;deadline=time.monotonic()+remaining()-600
                while process.poll() is None:
                    if cancelled.is_set() or time.monotonic()>=deadline:
                        if not terminated:
                            process.terminate();terminated=True;term_at=time.monotonic()
                        elif time.monotonic()-term_at>90:
                            # The allocation remains the final bound for a controller that cannot stop.
                            (Path(os.environ['MIRAGE_SESSION_DIR'])/'cancellation-unconfirmed').touch()
                            raise RuntimeError('Controller did not exit after termination; stop owned steps before service reuse')
                    time.sleep(.5)
                code=process.returncode
            e2e=time.monotonic()-begin
            service_idle=confirmed_idle()
            (root/(label+'-metrics-after.txt')).write_text(service_metrics())
            row={'index':i,**target,'directory':str(root/label),'exit_status':code,
                 'started_epoch':begin_epoch,'finished_epoch':time.time(),'e2e_seconds':e2e,
                 'service_idle_after':service_idle}
            rows.append(row)
            write(root/'campaign.json',{'state':'running','completed':rows,'expected':len(protocol['targets']),
                'elapsed_seconds':time.monotonic()-started})
            print(json.dumps(row),flush=True)
            if not service_idle or cancelled.is_set():
                state='failed_or_incomplete';break
        else: state='complete' if all(r['exit_status']==0 for r in rows) else 'complete_with_failures'
    except BaseException:
        state='error'
        raise
    finally:
        stop.set();thread.join(timeout=10)
        write(root/'campaign.json',{'state':state,'completed':rows,'expected':len(protocol['targets']),
            'elapsed_seconds':time.monotonic()-started,'remaining_unattempted':len(protocol['targets'])-len(rows)})
    verify(args.protocol)
    return 0 if state=='complete' else 1


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--engine',type=Path,required=True);p.add_argument('--label',required=True)
    p.add_argument('--budget',type=float,default=180);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--generation-overlap',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--active-apis',type=int,default=1)
    p.add_argument('--generation-concurrency',type=int,default=2)
    for name in ('campaign','one'):
        p=sub.add_parser(name);p.add_argument('--protocol',type=Path,required=True)
        if name=='one':p.add_argument('--index',type=int,required=True)
    args=parser.parse_args()
    if args.command=='prepare': prepare(args);return 0
    if args.command=='campaign':return campaign(args)
    return one(args)


if __name__=='__main__':raise SystemExit(main())

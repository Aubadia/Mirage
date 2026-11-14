"""Frozen general-oracle validation. Run computational stages only through Slurm."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
# Frozen before development test results; no per-API expected gradients.
HELDOUT = (
 ('unary','torch.expm1','torch.expm1, x'),
 ('unary','torch.log1p','torch.log1p, x'),
 ('multi_input','torch.lerp','torch.lerp, x, y, 0.3'),
 ('multi_input','torch.addcmul','torch.addcmul, x, y, y'),
 ('reduction','torch.prod','torch.prod, x'),
 ('reduction','torch.logsumexp','torch.logsumexp, x, dim=0'),
 ('matrix','torch.baddbmm','torch.baddbmm, x, y, y'),
 ('matrix','torch.linalg.solve','torch.linalg.solve, x, y'),
)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def evaluator():
    path=ROOT/'benchmarks/pytorch_correctness/evaluate.py'
    # The existing benchmark evaluator imports sibling fixture modules.
    sys.path.insert(0, str(path.parent))
    spec=importlib.util.spec_from_file_location('oracle_benchmark_evaluator',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def heldout(output):
    import torch
    from mirage.runtime.stateful import execute
    from mirage.runtime.probes import Recorder
    from mirage.inspector.oracles import audit_execution
    rows=[]
    for family,api,call in HELDOUT:
        values='[[[2.0,0.2],[0.1,1.5]]]' if api=='torch.baddbmm' else '[[2.0,0.2],[0.1,1.5]]'
        source=f'''import torch

def build(device,seed):
    x=torch.tensor({values},dtype=torch.float64,device=device,requires_grad=True)
    y=torch.tensor({values},dtype=torch.float64,device=device,requires_grad=True)
    return {{'inputs':{{'x':x,'y':y}},'parameters':{{}}}}

def run(ctx,probe):
    x=ctx['inputs']['x']
    y=ctx['inputs']['y']
    out=probe.call({call})
    out.sum().backward()
    return {{'output':out,'gradient':x.grad}}
'''
        folder=output/api;folder.mkdir(parents=True)
        (folder/'program.py').write_text(source)
        config=dict(api=api,device='cpu',seed=0,requires_backward=True)
        record=execute(source,config,Recorder(folder/'trace.jsonl'))
        audit=audit_execution(record,budget_seconds=30)
        record['gradient_audit']=audit;torch.save(record,folder/'record.pt')
        write(folder/'audit.json',audit)
        rows.append(dict(api=api,family=family,valid=record['inspection']['valid'],audit=audit))
        print(api,audit['status'],audit.get('reason'),flush=True)
    families={r['family'] for r in rows if r['valid'] and r['audit']['status']=='agreement'}
    result={'records':rows,'families_with_agreement':sorted(families),
            'passed':len(families)==4 and not any(r['audit']['status']=='violation' for r in rows),
            'scope':'Frozen held-out APIs; successful numerical checks establish sampled local consistency only.'}
    write(output/'summary.json',result)
    return 0 if result['passed'] else 1


def candidate(output, original):
    import torch
    from mirage.artifacts.bundle import snapshot_engine, inventory
    from mirage.runtime.process import run_worker
    from mirage.workflow import replay_stateful
    from mirage.backends.pytorch import PyTorchBackend
    engine=output/'engine';snapshot_engine(engine)
    case=output/'candidate';case.mkdir()
    source=(original/'program.py').read_text();(case/'program.py').write_text(source)
    old=json.loads((original/'stateful-manifest.json').read_text())
    config=dict(old['configuration'],numerical_audit=True)
    process=run_worker(dict(mode='execute_stateful_pair',source=source,config=config,
                           engine=str(engine),audit_budget_seconds=30),case/'execution',180)
    if process.outcome!='completed':
        write(output/'summary.json',{'passed':False,'process':process.__dict__});return 1
    summary=json.loads((case/'execution/summary.json').read_text())
    checks=[];module=evaluator()
    for member in ('source','followup'):
        record=torch.load(case/'execution'/f'{member}.pt',map_location='cpu',weights_only=True)
        previous=torch.load(original/'execution'/f'{member}.pt',map_location='cpu',weights_only=True)
        same=PyTorchBackend().compare(record['initial'],previous['initial'],'exact')['equal']
        observed=PyTorchBackend().compare(record['observations'],previous['observations'],'exact')['equal']
        audit=module.stateful_independent_audit('204407',record)
        checks.append(dict(member=member,initial_equal=same,observations_equal=observed,audit=audit,
                           attribution=module.stateful_attribution('204407',record,audit)))
    manifest={'format':'stateful-1-replay','source':'program.py','configuration':config,
              'engine':str(engine),'engine_inventory':inventory(engine),'files':inventory(case),
              'initial_reference':'execution/source.pt','expected_verdict':summary['verdict']}
    write(case/'stateful-manifest.json',manifest)
    replay=replay_stateful(case/'stateful-manifest.json',timeout=180)
    replay_checks=[]
    for member in ('source','followup'):
        record=torch.load(Path(replay['directory'])/f'{member}.pt',map_location='cpu',weights_only=True)
        audit=module.stateful_independent_audit('204407',record)
        replay_checks.append(dict(member=member,audit=audit,
            attribution=module.stateful_attribution('204407',record,audit)))
    passed=(summary['verdict']=='suspected_failure' and replay['replay_valid']
            and all(c['initial_equal'] and c['observations_equal'] for c in checks)
            and any(c['attribution']['benchmark_reproduction'] for c in checks)
            and any(c['attribution']['benchmark_reproduction'] for c in replay_checks))
    write(output/'summary.json',dict(passed=bool(passed),checks=checks,replay_checks=replay_checks,
        execution=summary,replay=replay,
        scope='Re-audit of a historical generated candidate under a new frozen engine; not a new search or a change to historical 0/10. Numerical corroboration, no API derivative used.'))
    print('candidate automatic numerical corroboration',passed,flush=True)
    return 0 if passed else 1


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=('regression','heldout','candidate','fixtures'),required=True)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--candidate',type=Path)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_STEP_ID'):
        p.error('Run through srun inside a Slurm allocation')
    a.output.mkdir(parents=True,exist_ok=False)
    if a.stage=='heldout':
        with (a.output/'cuda-gradient.log').open('w') as log:
            check=subprocess.run([sys.executable,'-m','unittest',
                'test_numerical_gradients.NumericalGradientTests.test_cuda_reconstruction_preserves_device_and_normal_gradient',
                '-v'],cwd=ROOT/'tests',stdout=log,stderr=subprocess.STDOUT)
        if check.returncode:return check.returncode
        return heldout(a.output)
    if a.stage=='candidate':
        if not a.candidate:p.error('--candidate required')
        return candidate(a.output,a.candidate)
    if a.stage=='fixtures':
        with (a.output/'cuda-evidence.log').open('w') as log:
            check=subprocess.run([sys.executable,'-m','unittest',
                'test_stateful.CapturedEvidenceTests','-v'],cwd=ROOT/'tests',
                stdout=log,stderr=subprocess.STDOUT)
        if check.returncode:return check.returncode
        module=evaluator()
        # The existing driver owns its output creation.
        return module.stateful_integration(a.output/'fixed',180)
    patterns=('test_numerical_gradients.py','test_oracle_routing.py','test_stateful.py',
              'test_capabilities.py','test_adaptive.py','test_pytorch_benchmark.py')
    rows=[]
    for pattern in patterns:
        started=time.monotonic()
        with (a.output/(pattern+'.log')).open('w') as log:
            run=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests'),'-p',pattern,'-v'],stdout=log,stderr=subprocess.STDOUT)
        rows.append(dict(pattern=pattern,exit_status=run.returncode,seconds=time.monotonic()-started))
        write(a.output/'summary.json',dict(records=rows,passed=all(r['exit_status']==0 for r in rows)))
        print(rows[-1],flush=True)
    return int(any(r['exit_status'] for r in rows))

if __name__=='__main__':raise SystemExit(main())

"""Phase 2 integration and independent Phase 3 evaluation; never imported by generation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from mirage.contracts.programs import RunOptions
from mirage.runtime.process import run_worker
from mirage.artifacts.bundle import snapshot_engine, inventory
from mirage.workflow import run_properties


HERE=Path(__file__).resolve().parent

def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def integration(root, timeout):
    from cases import INTEGRATION_CASES
    root.mkdir(parents=True,exist_ok=False)
    snapshot_engine(root/'engine')
    rows=[]
    for issue,(api,device,family,source) in INTEGRATION_CASES.items():
        directory=root/issue
        config=dict(api=api,device=device,family=family,representation='original',seed=0,
                    backend='inductor' if api=='torch.std' else 'eager')
        process=run_worker(dict(mode='execute_property',source=source,config=config,engine=str(root/'engine')),
                           directory,timeout)
        write(root/(issue+'.sha256.json'),inventory(directory))
        outcome=json.loads((directory/'summary.json').read_text()) if (directory/'summary.json').exists() else {}
        row=dict(issue=issue,api=api,process=process.outcome,verdict=outcome.get('verdict'),
                 api_executed=outcome.get('api_executed',False),reason=outcome.get('reason',process.error),
                 controls_agree=outcome.get('control_differences')==[],directory=str(directory))
        row['detected']=process.outcome=='completed' and row['verdict']=='suspected_failure' and row['api_executed'] and row['controls_agree']
        rows.append(row);write(root/'summary.json',dict(records=rows,detected=sum(r['detected'] for r in rows),total=len(INTEGRATION_CASES)))
        print(issue,row,flush=True)
    write(root/'engine-sha256.json',inventory(root/'engine'))
    return 0 if all(r['detected'] for r in rows) else 1


def trigger(issue, record):
    """Post-hoc structural trigger evidence, not ground-truth failure labels."""
    values=list(record.get('actual_metadata',{}).values())
    if not values:
        values=[v for c in record.get('target_calls',[]) for v in c.get('metadata',{}).values()]
    if issue=='203078':
        indices=[v for v in values if v.get('stride')==[1,0] and v.get('dtype')=='torch.int64']
        return any(any(len(v.get('shape',[]))==2 and v['shape'][1]>i['shape'][1] and v.get('dtype','').startswith('torch.float') for v in values) for i in indices)
    if issue=='204407': return any(v.get('neg') for v in values) and record.get('backward_executed',False)
    if issue=='200599': return any(len(v.get('shape',[]))>=3 and v['shape'][-3]==1 and v.get('stride',[0])[-3] != v['shape'][-2]*v['shape'][-1] for v in values if 'stride' in v)
    if issue=='202706': return record.get('depth',0)>0 and record.get('api_executed',False)
    if issue=='181723': return record.get('microbatches',0)>1 and record.get('backward_executed',False)
    if issue=='204033': return record.get('predicates',{}).get('returned_alias',False) and bool(record.get('compiled_graphs')) and any(v.get('neg') for v in record.get('reference_metadata',{}).values())
    if issue=='204365': return any(v.get('stride')==[2] for v in values) and any(v.get('layout')=='torch._mkldnn' for v in values)
    if issue=='201268': return record.get('actual_metadata',{}).get("(1, 'out')",{}).get('values',{}).get('stride',[1]) != [1]
    if issue=='197089':
        ref=record.get('reference',{}).get('output')
        return ref is not None and bool(ref.abs()>0) and bool(ref.double().square().float()==0) and record.get('config',{}).get('backend')=='inductor' and record.get('backward_executed',False)
    if issue=='200613':
        # Upstream explicitly includes negative views and sliced temporary-copy paths.
        src=record.get('actual_metadata',{}).get('(0, 1)',{})
        dst=record.get('actual_metadata',{}).get('(0, 0)',{})
        return (src.get('device')=='cpu' and dst.get('device','').startswith('cuda') and
                bool(src.get('conj') or src.get('neg')) and
                (src.get('shape')!=dst.get('shape') or src.get('contiguous') is False))
    return False


def summarize(root):
    import torch
    catalog=json.loads((HERE/'candidates.json').read_text())
    targets={c['id']:c for c in catalog['candidates'] if c['id'] in catalog['selected']}
    reports=[]
    for issue,c in targets.items():
        p=root/issue/'summary.json'
        if not p.exists(): continue
        run=json.loads(p.read_text());rows=run['records'];detections=[];triggers=0
        generated=sum(r.get('generated',False) for r in rows)
        valid=sum(r.get('contract_valid',False) for r in rows)
        executable=sum(r.get('executable',False) and r.get('api_executed',False) for r in rows)
        adjudications=[]
        for row in rows:
            evidence=Path(row.get('directory',''))/'execution/result.pt'
            if not evidence.exists(): continue
            record=torch.load(evidence,map_location='cpu',weights_only=True)
            reached=trigger(issue,record);triggers+=bool(reached)
            matched=reached and row.get('verdict')=='suspected_failure'
            if matched: detections.append(row['finished_seconds'])
            adjudications.append(dict(attempt=row['attempt'],structural_trigger=reached,
                benchmark_consistent_detection=bool(matched),independent_ground_truth='unadjudicated',
                note='Structural conditions alone do not establish a positive example or common root cause.'))
        report=dict(issue=issue,api=c['api'],generated=generated,contract_valid=valid,executable=executable,
                    valid_test_rate=valid/generated if generated else None,
                    executable_test_rate=executable/valid if valid else None,
                    structural_trigger_cases=triggers,benchmark_consistent_detections=len(detections),
                    time_to_first_detection_seconds=min(detections) if detections else None,
                    first_detection_censored=not bool(detections),false_positive_rate=None,
                    correctness_violation_detection_rate=None,
                    reason_unmeasured_rates='No independently adjudicated positive/negative generated population yet',
                    elapsed_seconds=run.get('elapsed_seconds'),stop_reason=run.get('stop_reason'),adjudications=adjudications)
        reports.append(report)
    output=dict(reports=reports,target_count=len(targets),completed_targets=len(reports),
                benchmark_consistent_issue_rate=sum(r['benchmark_consistent_detections']>0 for r in reports)/len(targets),
                false_positive_rate=None,independently_confirmed_issue_reproduction_rate=None)
    write(root/'evaluation.json',output)
    return output


def experiment(root, budget, phase2):
    approval=json.loads((phase2/'summary.json').read_text())
    if approval.get('detected')!=approval.get('total') or approval.get('total')!=10:
        raise ValueError('Phase 2 must detect all ten benchmark fixtures before blind evaluation')
    catalog=json.loads((HERE/'candidates.json').read_text())
    root.mkdir(parents=True,exist_ok=False)
    snapshot_engine(root/'frozen-engine')
    frozen=inventory(root/'frozen-engine')
    validated=json.loads((phase2/'engine-sha256.json').read_text())
    if {k:v for k,v in validated.items() if k.endswith('.py')}!=frozen:
        raise ValueError('Experiment engine differs from validated Phase 2 engine')
    write(root/'engine-sha256.json',frozen)
    write(root/'protocol.json',dict(budget_seconds_per_api=budget,provider='vllm',
                                  phase2=str(phase2),issue_data_in_prompts=False,deterministic_inputs=True))
    for issue in catalog['selected']:
        c=next(c for c in catalog['candidates'] if c['id']==issue)
        # Only the API/device crosses into production generation. Issue data stays here.
        output=run_properties(RunOptions(str(root/issue),'vllm',cases=100000,time_limit=budget,
                                       worker_timeout=min(120,budget),device=c['device'],strategy='properties'),(c['api'],))
        print(issue,len(output['records']),output['stop_reason'],flush=True)
        summarize(root)
        if output['stop_reason']=='provider_cancellation_unconfirmed': return 2
    summarize(root)
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('validate','experiment','report','audit'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase2',type=Path)
    p.add_argument('--budget',type=int,default=300)
    p.add_argument('--timeout',type=int,default=180)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_STEP_ID'):
        p.error('Execute through srun in Slurm')
    if a.mode=='audit':
        output=independent_audit(a.output)
        print(json.dumps({k:v for k,v in output.items() if k!='records'},indent=2));return 0
    if a.mode=='validate': return integration(a.output,a.timeout)
    if a.mode=='experiment':
        if a.phase2 is None or a.budget<=0: p.error('--phase2 and positive --budget required')
        return experiment(a.output,a.budget,a.phase2)
    print(json.dumps(summarize(a.output),indent=2));return 0




# Independent post-hoc audit. These routines do not run in the generation path.
def independent_expected(issue, ctx, source):
    import ast
    import itertools
    import math
    from fractions import Fraction
    import torch

    args,kw=ctx.get('args',()),ctx.get('kwargs',{})
    def values(t):
        return t.detach().resolve_conj().resolve_neg().cpu().tolist()
    def flat(v):
        return [x for child in v for x in flat(child)] if isinstance(v,list) else [v]
    def tensor(v,like):
        return torch.tensor(v,dtype=like.dtype)
    if issue=='204407':
        x=args[0]
        if x.ndim!=2 or x.shape[0]!=x.shape[1] or x.shape[0]>8 or x.dtype not in (torch.float32,torch.float64):
            raise ValueError('Exact rational determinant audit supports real matrices up to 8x8')
        a=values(x); n=len(a)
        rows=[[Fraction(v) for v in row]+[Fraction(int(i==j)) for j in range(n)] for i,row in enumerate(a)]
        det=Fraction(1)
        for col in range(n):
            pivot=next((i for i in range(col,n) if rows[i][col]),None)
            if pivot is None: raise ValueError('Singular input')
            if pivot!=col: rows[col],rows[pivot]=rows[pivot],rows[col];det=-det
            d=rows[col][col];det*=d;rows[col]=[v/d for v in rows[col]]
            for i in range(n):
                if i!=col:
                    c=rows[i][col];rows[i]=[v-c*w for v,w in zip(rows[i],rows[col])]
        inverse=[r[n:] for r in rows]
        condition=max(sum(abs(v) for v in row) for row in a)*max(sum(abs(float(v)) for v in row) for row in inverse)
        if condition>100: raise ValueError('Independent determinant condition bound exceeded')
        gradient=[[float(inverse[j][i]) for j in range(n)] for i in range(n)]
        return dict(output=tensor(math.log(abs(float(det))),x),gradient=tensor(gradient,x)), 'Python Fraction elimination; inverse-transpose derivative'
    if issue=='197089':
        x=args[0]
        if x.dtype not in (torch.float32,torch.float64) or len(args)>1 or kw.get('dim') is not None or kw.get('keepdim',False):
            raise ValueError('Audit requires scalar all-element reduction')
        v=flat(values(x));den=len(v)-kw.get('correction',1 if kw.get('unbiased',True) else 0)
        if den<=0: raise ValueError('Nonpositive degrees of freedom')
        mean=math.fsum(v)/len(v);center=[a-mean for a in v]
        std=math.hypot(*center)/math.sqrt(den)
        if not std>0: raise ValueError('Zero standard deviation')
        gradient=tensor([a/(den*std) for a in center],x).reshape(x.shape)
        return dict(output=tensor(std,x),gradient=gradient),'Python math.fsum/hypot scalar formula and analytic derivative'
    if issue=='202706':
        allowed={'dict','int','str','torch.device','torch.ones','torch.zeros','torch.arange',
                 'torch.tensor','torch.linspace','torch.eye','torch.full','torch.complex'}
        methods={'view','reshape','clone','to','contiguous','detach','as_strided','unsqueeze',
                 'squeeze','expand','transpose','conj','resolve_conj','float','double','requires_grad_'}
        for node in ast.walk(ast.parse(source)):
            legal_call=not isinstance(node,ast.Call) or ast.unparse(node.func) in allowed or isinstance(node.func,ast.Attribute) and node.func.attr in methods
            if isinstance(node,(ast.With,ast.AsyncWith)) or not legal_call:
                raise ValueError('Builder may alter ambient context; independent entry-state audit unavailable')
        device=kw.get('device_type',args[0] if args else 'cpu')
        if device not in ('cpu','cuda'): raise ValueError('Unknown initial autocast dtype')
        return dict(grad=True,inference=False,amp=False,dtype='torch.bfloat16' if device=='cpu' else 'torch.float16',cache=True),'Fresh worker default thread-local state; public context restoration contract'
    if issue=='203078':
        dst,dim,index,src=args
        if dst.dtype not in (torch.float32,torch.float64) or dst.numel()>8192 or src.numel()>8192:
            raise ValueError('Integer-exact scatter audit bounds')
        base,source_values=flat(values(dst)),flat(values(src))
        if any(not float(v).is_integer() or abs(v)>4096 for v in base+source_values):
            raise ValueError('Scatter audit restricted to exactly representable integers')
        dim%=dst.ndim; expected=list(base)
        def offset(point,shape):
            result=0
            for i,n in zip(point,shape): result=result*n+i
            return result
        ids=flat(values(index))
        for linear,point in enumerate(itertools.product(*(range(n) for n in index.shape))):
            target=list(point);target[dim]=ids[linear]
            expected[offset(target,dst.shape)]+=source_values[offset(point,src.shape)]
        if max(map(abs,expected),default=0)>=2**24: raise ValueError('Exact float32 sum bound exceeded')
        return tensor(expected,dst).reshape(dst.shape),'Python integer scatter accumulation, exactly representable in target dtype'
    if issue=='200613':
        dst,src=args[:2]
        if dst.dtype!=src.dtype or dst.numel()>8192:
            raise ValueError('Copy audit requires equal dtype and bounded size')
        source_values=flat(values(src));shape=(1,)*(dst.ndim-src.ndim)+tuple(src.shape)
        if len(shape)!=dst.ndim or any(a not in (1,b) for a,b in zip(shape,dst.shape)):
            raise ValueError('Invalid broadcast')
        expected=[]
        for point in itertools.product(*(range(n) for n in dst.shape)):
            offset=0
            for p,n in zip(point,shape): offset=offset*n+(0 if n==1 else p)
            expected.append(source_values[offset])
        return tensor(expected,dst).reshape(dst.shape),'Python broadcast indexing of logical source values, equal dtype'
    if issue=='204365':
        def operand(n,name,default=None): return args[n] if len(args)>n else kw.get(name,default)
        if operand(5,'training',False): raise ValueError('Audit supports inference statistics only')
        x,mean,var=args[:3];weight=operand(3,'weight');bias=operand(4,'bias');eps=operand(7,'eps',1e-5)
        if mean is None or var is None or x.numel()>8192 or x.dtype not in (torch.float32,torch.float64): raise ValueError('Unsupported inference operands')
        xv=flat(values(x.to_dense() if x.is_mkldnn else x));mv=values(mean);vv=values(var)
        wv=values(weight) if weight is not None else [1]*len(mv)
        bv=values(bias) if bias is not None else [0]*len(mv)
        spatial=math.prod(x.shape[2:]);channels=x.shape[1]
        expected=[]
        for i,a in enumerate(xv):
            c=(i//spatial)%channels
            expected.append((a-mv[c])/math.sqrt(vv[c]+eps)*wv[c]+bv[c])
        return tensor(expected,x).reshape(x.shape),'Python scalar inference normalization with logical affine values'
    if issue=='201268':
        a,b=args[:2]
        if not isinstance(a,torch.Tensor) or not isinstance(b,torch.Tensor) or a.shape!=b.shape or a.dtype!=b.dtype or a.numel()>8192:
            raise ValueError('Addition audit supports bounded equal-shape/equal-dtype operands')
        if a.dtype not in (torch.float32,torch.float64,torch.int32,torch.int64) or kw.get('alpha',1) not in (-2,-1,0,1,2):
            raise ValueError('Addition precision/alpha outside independent audit')
        def dense(t):
            if t.layout==torch.sparse_csr:
                result=[0]*t.numel();crow=values(t.crow_indices());col=values(t.col_indices());v=values(t.values())
                for row in range(t.shape[0]):
                    for k in range(crow[row],crow[row+1]): result[row*t.shape[1]+col[k]]+=v[k]
                return result
            if t.layout!=torch.strided: raise ValueError('Unsupported sparse layout')
            return flat(values(t))
        expected=[x+kw.get('alpha',1)*y for x,y in zip(dense(a),dense(b))]
        return tensor(expected,a).reshape(a.shape),'Python scalar addition; CSR indices expanded without sparse addition kernel'
    raise ValueError('No independent post-hoc oracle implemented for this family')


def independent_audit(root):
    """Assess saved observations, without rerunning any target or consulting its verdict."""
    import math
    import torch
    from mirage.rules.properties import decode_initial
    def equal(a,b,policy):
        if isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor):
            if a.shape!=b.shape or a.dtype!=b.dtype: return False
            av=a.detach().cpu().flatten().tolist();bv=b.detach().cpu().flatten().tolist()
            if not all(math.isfinite(abs(x)) and math.isfinite(abs(y)) for x,y in zip(av,bv)):
                raise ValueError('Nonfinite observations outside independent audit')
            errors=[abs(x-y) for x,y in zip(av,bv)]
            allowed=[policy['atol']+policy['rtol']*abs(y) for y in bv]
            if all(e<=t for e,t in zip(errors,allowed)): return True
            # Do not turn near-threshold rounding into an independently confirmed defect.
            eps=torch.finfo(b.real.dtype if b.is_complex() else b.dtype).eps if b.is_floating_point() or b.is_complex() else 0.
            if policy.get('exact'): eps=0.
            margin=256*eps*max(max(map(abs,bv),default=0.),policy.get('floor_scale',0.))*policy.get('condition_bound',1)
            return False if any(e>t+margin for e,t in zip(errors,allowed)) else None
        if isinstance(a,dict) and isinstance(b,dict):
            if a.keys()!=b.keys(): return False
            results=[equal(a[k],b[k],policy) for k in a]
            return False if False in results else None if None in results else True
        return type(a) is type(b) and a==b
    rows=[]
    for run in sorted(root.iterdir()):
        summary=run/'summary.json'
        if not run.is_dir() or not summary.exists(): continue
        data=json.loads(summary.read_text())
        for item in data.get('records',[]):
            directory=Path(item['directory'])/'execution'; evidence=directory/'result.pt'
            row=dict(issue=run.name,attempt=item['attempt'],label='unadjudicated',reported=item.get('verdict')=='suspected_failure')
            try:
                if not evidence.exists(): raise ValueError('No completed runtime evidence')
                record=torch.load(evidence,map_location='cpu',weights_only=True)
                if not record.get('api_executed') or 'observables' not in record: raise ValueError('No target observations')
                ctx=decode_initial(torch.load(directory/'initial.pt',map_location='cpu',weights_only=True)['context'])
                expected,method=independent_expected(run.name,ctx,(directory.parent/'program.py').read_text())
                # These policies match the documented screening criterion, not the observed verdict.
                policy=dict(atol=0.,rtol=1e-4) if run.name in ('204407','197089') else dict(atol=1e-5,rtol=1e-4)
                if run.name in ('202706','200613'): policy=dict(atol=0.,rtol=0.,exact=True)
                if run.name=='204407': policy.update(condition_bound=100,floor_scale=1.)
                correct=equal(record['observables'],expected,policy)
                if correct is None: raise ValueError('Difference lies inside conservative rounding uncertainty band')
                row.update(label='negative' if correct else 'positive',method=method,tolerance=policy,
                           structural_trigger=bool(trigger(run.name,record)))
                row['classification']=('FP' if row['reported'] else 'TN') if correct else ('TP' if row['reported'] else 'FN')
                row['finished_seconds']=item['finished_seconds']
            except Exception as error:
                row['reason']=f'{type(error).__name__}: {error}'
            rows.append(row)
    counts={name:sum(r.get('classification')==name for r in rows) for name in ('TP','FP','TN','FN')}
    positives=counts['TP']+counts['FN']; negatives=counts['FP']+counts['TN']
    output=dict(records=rows,counts=counts,adjudicated=sum(counts.values()),total_candidates=len(rows),
                false_positive_rate=counts['FP']/negatives if negatives else None,
                correctness_violation_detection_rate=counts['TP']/positives if positives else None,
                scope='Post-hoc independently computed reference subset; unadjudicated cases excluded; not a population-wide estimate')
    write(root/'independent-audit.json',output)
    return output


if __name__=='__main__': raise SystemExit(main())

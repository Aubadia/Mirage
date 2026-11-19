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
    p.add_argument('mode',choices=('validate','experiment','report','audit','validate-stateful','audit-stateful-fixtures','pilot-stateful','edges-stateful','heldout-stateful','freeze-stateful','experiment-stateful','report-stateful'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase2',type=Path)
    p.add_argument('--gate',type=Path)
    p.add_argument('--pilot',type=Path)
    p.add_argument('--edges',type=Path)
    p.add_argument('--heldout',type=Path)
    p.add_argument('--regression',type=Path)
    p.add_argument('--generation-budget',type=int)
    p.add_argument('--generation-cases',type=int)
    p.add_argument('--budget',type=int,default=300)
    p.add_argument('--timeout',type=int,default=180)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_STEP_ID'):
        p.error('Execute through srun in Slurm')
    if a.mode=='freeze-stateful': return freeze_stateful(a)
    if a.mode=='experiment-stateful': return experiment_stateful(a.output,a.budget,a.gate)
    if a.mode=='report-stateful':
        print(json.dumps(report_stateful(a.output),indent=2));return 0
    if a.mode in ('pilot-stateful','edges-stateful','heldout-stateful'):
        return stateful_model_checks(a.output,a.mode,a.generation_budget,a.generation_cases)
    if a.mode=='audit':
        output=independent_audit(a.output)
        print(json.dumps({k:v for k,v in output.items() if k!='records'},indent=2));return 0
    if a.mode=='audit-stateful-fixtures':
        outcome=stateful_fixture_audit(a.output);print(json.dumps(outcome,indent=2))
        return 0 if outcome['all_ten_independently_corroborated'] else 1
    if a.mode=='validate-stateful': return stateful_integration(a.output,a.timeout)
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


def audit_stateful_arithmetic(record):
    """Independent bounded rational arithmetic audit; never reads production labels.

    This supports an elementwise arithmetic semantic family, including unequal
    broadcastable shapes. Unknown operations/effects remain explicitly unknown.
    """
    import itertools
    import math
    from fractions import Fraction
    import torch
    operation = record.get('config', {}).get('api', '').rsplit('.', 1)[-1]
    operators = {'add': lambda a,b:a+b, 'sub': lambda a,b:a-b,
                 'mul': lambda a,b:a*b, 'div': lambda a,b:a/b}
    unknown = lambda reason: {'label': 'unknown', 'reason': reason, 'method': 'rational_elementwise_v1'}
    if operation not in operators: return unknown('outside supported arithmetic family')
    labels = []
    for call in record.get('target_calls', []):
        args, kwargs = call['inputs']['args'], call['inputs']['kwargs']
        if len(args) != 2 or set(kwargs) - ({'alpha'} if operation in ('add', 'sub') else set()):
            return unknown('unsupported binding/options')
        output = call['output']
        if (not isinstance(output, torch.Tensor) or output.layout != torch.strided
                or output.numel() > 4096 or not output.is_floating_point()):
            return unknown('unsupported output layout/size')
        def operand(x):
            if isinstance(x, torch.Tensor):
                if x.layout != torch.strided or x.is_complex(): raise ValueError('unsupported input representation')
                return tuple(x.shape), x.detach().cpu().tolist()
            if type(x) not in (int, float, bool): raise ValueError('unsupported scalar type')
            return (), x
        try:
            operands = [operand(x) for x in args]
            rank = max(len(shape) for shape, _ in operands)
            aligned = [(1,)*(rank-len(shape)) + shape for shape, _ in operands]
            shape = tuple(b if a == 1 else a for a,b in zip(*aligned))
            if any(a != b and a != 1 and b != 1 for a,b in zip(*aligned)):
                return unknown('inputs cannot broadcast')
            if tuple(output.shape) != shape: return {'label':'positive', 'reason':'incorrect broadcast output shape', 'method':'rational_elementwise_v1'}
            alpha = kwargs.get('alpha', 1)
            if type(alpha) not in (int, float) or not math.isfinite(alpha): return unknown('unsupported alpha')
            alpha = Fraction(alpha)
            actual = output.detach().cpu().reshape(-1).tolist()
            for index, observed in zip(itertools.product(*(range(n) for n in shape)), actual):
                values = []
                for dims, data in operands:
                    indices = index[rank-len(dims):] if dims else ()
                    for coordinate, extent in zip(indices, dims): data = data[0 if extent == 1 else coordinate]
                    if not math.isfinite(data): return unknown('nonfinite input')
                    values.append(Fraction(data))
                if operation in ('add', 'sub'): values[1] *= alpha
                reference = float(operators[operation](*values))
                if not math.isfinite(reference) or not math.isfinite(observed): return unknown('nonfinite reference/observation')
                scale = max(abs(reference), abs(observed))
                error = 0. if scale == 0 else abs(reference/scale - observed/scale)
                epsilon = torch.finfo(output.dtype).eps if output.is_floating_point() else 0.
                if error <= 4*epsilon: labels.append('negative')
                elif error > 64*epsilon: labels.append('positive')
                else: return unknown('rounding uncertainty band')
        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as error:
            return unknown(str(error))
    if not labels: return unknown('no audited target elements')
    return {'label': 'positive' if 'positive' in labels else 'negative',
            'reason': 'independent rational arithmetic with rounding uncertainty',
            'method': 'rational_elementwise_v1', 'audited_elements': len(labels)}


def stateful_fixture_programs():
    """Only benchmark fixtures: no caller in production generation imports this function."""
    import ast
    from cases import INTEGRATION_CASES
    programs = {}
    for issue,(api,device,family,source) in INTEGRATION_CASES.items():
        if family not in ('materialization','gradient_reference'): continue
        tree = ast.parse(source)
        build = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'build')
        returned = build.body.pop()
        if not isinstance(returned,ast.Return): raise ValueError('Fixture must end with a return')
        build.body.append(ast.Assign([ast.Name('payload',ast.Store())],returned.value))
        if issue in ('204407','197089'):
            tail = """
x = payload['args'][0].detach().requires_grad_()
payload['args'] = (x,)
return {'inputs':payload,'parameters':{'x':x}}
"""
        else:
            tail = "return {'inputs':payload,'parameters':{}}"
        build.body.extend(ast.parse(tail).body)
        if issue == '197089':
            body = """
def run(ctx, probe):
    def operation(x):
        return torch.std(x, correction=0)
    compiled = torch.compile(operation,backend='inductor',fullgraph=True)
    output = compiled(ctx['inputs']['args'][0])
    output.backward()
    return {'output':output,'gradient':ctx['parameters']['x'].grad}
"""
        elif issue == '204407':
            body = f"""
def run(ctx, probe):
    output = probe.call({api},*ctx['inputs']['args'],**ctx['inputs']['kwargs'])
    output.logabsdet.backward()
    return {{'output':output.logabsdet,'gradient':ctx['parameters']['x'].grad}}
"""
        else:
            body = f"""
def run(ctx, probe):
    output = probe.call({api},*ctx['inputs']['args'],**ctx['inputs']['kwargs'])
    return {{'output':output}}
"""
        tree.body.extend(ast.parse(body).body)
        programs[issue] = (api,device,ast.unparse(ast.fix_missing_locations(tree)),issue in ('204407','197089'))
    programs['202706'] = ('torch.autocast','cpu', '''
import torch
def build(device,seed):
    return {'inputs':{},'parameters':{}}
def run(ctx,probe):
    mode = probe.call(torch.autocast,'cpu',dtype=torch.bfloat16)
    def fresh(depth):
        with torch.autocast('cpu',dtype=torch.bfloat16):
            if depth: fresh(depth-1)
    @mode
    def nested(depth):
        if depth: nested(depth-1)
    fresh(1)
    nested(1)
    return {}
''',False)
    programs['204033'] = ('torch.compile','cpu','''
import torch
def build(device,seed):
    return {'inputs':{'x':torch.arange(1.,5.)},'parameters':{}}
def run(ctx,probe):
    def operation(x):
        base = torch.complex(x,x+1)
        view = base.conj().imag
        view.add_(3)
        return base,view
    compiled = probe.call(torch.compile,operation,backend='inductor',fullgraph=True)
    base,view = compiled(ctx['inputs']['x'])
    view.add_(100.)
    return {'base':base,'view':view}
''',False)
    programs['181723'] = ('torch.cuda.make_graphed_callables','cuda','''
import torch
def build(device,seed):
    class Scale(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(1.,5.,device=device))
        def forward(self,x):
            return x*self.weight
    module = Scale()
    x1 = torch.tensor([[1.,2.,3.,4.],[2.,3.,4.,5.]],device=device)
    x2 = torch.tensor([[10.,20.,30.,40.],[1.,2.,3.,4.]],device=device)
    return {'inputs':{'batches':(x1,x2)},'parameters':{'weight':module.weight},'modules':{'model':module}}
def run(ctx,probe):
    model = ctx['modules']['model']
    a,b = ctx['inputs']['batches']
    captured = probe.call(torch.cuda.make_graphed_callables,model,(torch.ones_like(a),))
    model.weight.grad = None
    captured(a).sum().backward()
    captured(b).sum().backward()
    return {'gradient':model.weight.grad}
''',True)
    return programs


def stateful_integration(root,timeout):
    root.mkdir(parents=True,exist_ok=False)
    snapshot_engine(root/'engine')
    rows = []
    for issue,(api,device,source,backward) in stateful_fixture_programs().items():
        directory = root/issue
        config = {'api':api,'device':device,'seed':0,'requires_backward':backward,'canonical_reference':True}
        process = run_worker({'mode':'execute_stateful_pair','source':source,'config':config,'engine':str(root/'engine')},directory,timeout)
        outcome = json.loads((directory/'summary.json').read_text()) if (directory/'summary.json').exists() else {}
        row = {'issue':issue,'api':api,'process':process.outcome,'verdict':outcome.get('verdict'),
               'reason':outcome.get('reason',process.error),'missing':outcome.get('missing'),
               'api_executed':outcome.get('api_executed',False),'directory':str(directory)}
        row['detected'] = row['process']=='completed' and row['verdict']=='suspected_failure' and row['api_executed']
        rows.append(row)
        write(directory/'source.py.json',{'source':source,'config':config})
        write(root/'summary.json',{'records':rows,'detected':sum(r['detected'] for r in rows),'total':10,
                                  'controls_independently_audited':False,'gate_complete':False})
        print(json.dumps(row),flush=True)
    write(root/'engine-sha256.json',inventory(root/'engine'))
    audit=stateful_fixture_audit(root)
    return 0 if all(r['detected'] for r in rows) and audit['all_ten_independently_corroborated'] else 1


def stateful_fixture_audit(root):
    """Independent expected values for fixed fixtures, never production generation."""
    import itertools
    import math
    from fractions import Fraction
    import torch
    summary = json.loads((root/'summary.json').read_text())
    def same(actual, expected, exact=False):
        if isinstance(actual,torch.Tensor) and isinstance(expected,torch.Tensor):
            if actual.shape != expected.shape or actual.dtype != expected.dtype: return False
            aa,bb = actual.detach().cpu().reshape(-1).tolist(),expected.detach().cpu().reshape(-1).tolist()
            epsilon = 0. if exact else (2**-52 if actual.dtype in (torch.float64,torch.complex128) else 2**-23)
            scale = max((abs(x) for x in bb),default=0.)
            return all(math.isfinite(abs(a)) and math.isfinite(abs(b)) and abs(a-b) <= 128*epsilon*scale for a,b in zip(aa,bb))
        if isinstance(actual,dict) and isinstance(expected,dict):
            return actual.keys()==expected.keys() and all(same(actual[k],expected[k],exact) for k in actual)
        return actual == expected
    def lu_expected(inputs):
        lu,piv,rhs = inputs['args']
        n = lu.shape[-1]
        if n>8 or lu.shape[-2]!=n: raise ValueError('bounded square linear-system reference only')
        shapes = [tuple(lu.shape[:-2]),tuple(piv.shape[:-1]),tuple(rhs.shape[:-2])]
        rank=max(map(len,shapes)); aligned=[(1,)*(rank-len(s))+s for s in shapes]
        batch=tuple(max(dim) for dim in zip(*aligned))
        arrays=[t.tolist() for t in (lu,piv,rhs)]
        def at(value,shape,point):
            for extent,index in zip(shape,point[rank-len(shape):] if shape else ()): value=value[0 if extent==1 else index]
            return value
        solutions=[]
        for point in itertools.product(*(range(size) for size in batch)):
            packed,pivots,b = [at(v,s,point) for v,s in zip(arrays,shapes)]
            lower=[[Fraction(1 if i==j else packed[i][j] if i>j else 0) for j in range(n)] for i in range(n)]
            upper=[[Fraction(packed[i][j] if i<=j else 0) for j in range(n)] for i in range(n)]
            a=[[sum(lower[i][k]*upper[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
            for i in reversed(range(n)): a[i],a[pivots[i]-1]=a[pivots[i]-1],a[i]
            augmented=[row+[Fraction(x) for x in rhsrow] for row,rhsrow in zip(a,b)]
            for column in range(n):
                pivot=next(i for i in range(column,n) if augmented[i][column])
                augmented[column],augmented[pivot]=augmented[pivot],augmented[column]
                divisor=augmented[column][column];augmented[column]=[x/divisor for x in augmented[column]]
                for i in range(n):
                    if i!=column:
                        coefficient=augmented[i][column]
                        augmented[i]=[x-coefficient*y for x,y in zip(augmented[i],augmented[column])]
            solutions.extend(float(x) for row in augmented for x in row[n:])
        return torch.tensor(solutions,dtype=rhs.dtype).reshape(*batch,n,rhs.shape[-1])
    audited=[]
    for row in summary['records']:
        issue=row['issue'];directory=Path(row['directory'])
        result={'issue':issue,'label':'unknown','controls_agree':False}
        try:
            source=torch.load(directory/'source.pt',map_location='cpu',weights_only=True)
            follow=torch.load(directory/'followup.pt',map_location='cpu',weights_only=True) if (directory/'followup.pt').exists() else None
            actual,control=source,follow
            if source.get('compiled_graphs'): actual,control=follow,source
            if issue=='202706':
                events=source['context_events']
                if len(events)!=4 or not all(e['balanced'] for e in events): raise ValueError('context fixture evidence incomplete')
                positive=any(e['before']!=e['after'] for e in events[2:])
                controls=all(e['before']==e['after'] for e in events[:2])
                method='Public restoration contract; independent fresh-instance scopes recorded before reused decoration'
            else:
                if actual is None or control is None: raise ValueError('paired evidence missing')
                inputs=actual['target_calls'][0]['inputs']
                if issue=='200599':
                    expected=lu_expected(inputs);a=actual['observations']['output'];c=control['observations']['output']
                    method='Python Fraction LU reconstruction, inverse pivot application and Gaussian elimination'
                elif issue=='204033':
                    x=actual['initial']['inputs']['x']
                    values=x.tolist()
                    expected={'base':torch.tensor([complex(v,v+1-3-100) for v in values],dtype=torch.complex64),
                              'view':torch.tensor([-(v+1-3-100) for v in values],dtype=x.dtype)}
                    a=actual['observations'];c=control['observations']
                    method='Scalar complex arithmetic and documented alias mutation propagation'
                elif issue=='181723':
                    batches=actual['initial']['inputs']['batches']
                    rows=[r for batch in batches for r in batch.tolist()]
                    expected={'gradient':torch.tensor([math.fsum(r[j] for r in rows) for j in range(len(rows[0]))],dtype=torch.float32)}
                    a=actual['observations'];c=control['observations']
                    method='Python sum of explicit batch values for the fixture linear model derivative'
                else:
                    expected,method=independent_expected(issue,inputs,'')
                    if issue in ('204407','197089'):
                        a=actual['observations'];c=control['observations']
                    else:
                        a=actual['observations']['output'];c=control['observations']['output']
                exact=issue in ('203078','201268','200613','204033','181723')
                controls=same(c,expected,exact);positive=not same(a,expected,exact)
            result.update(label='positive' if positive and controls else 'negative' if controls else 'unknown',
                          controls_agree=bool(controls),method=method,
                          production_reported=row.get('verdict')=='suspected_failure')
        except Exception as error:
            result['reason']=f'{type(error).__name__}: {error}'
        audited.append(result)
    passed=len(audited)==10 and all(r['label']=='positive' and r['controls_agree'] and r.get('production_reported') for r in audited)
    output={'records':audited,'all_ten_independently_corroborated':passed,
            'scope':'Fixed deterministic fixtures; not generated-case detection or false-positive estimates'}
    write(root/'independent-controls.json',output)
    summary.update(controls_independently_audited=passed,fixture_gate_complete=passed,gate_complete=False)
    write(root/'summary.json',summary)
    return output



def stateful_mechanism(issue, record):
    """Post-hoc structural eligibility, never a defect label or generation hint."""
    calls=record.get('target_calls',[])
    descriptors=[d for call in calls for d in call.get('metadata',[])]
    def prop(d,key): return d.get('properties',{}).get(key,False)
    def noncontiguous(d):
        shape,strides=d.get('shape',[]),d.get('strides')
        if strides is None or 0 in shape: return False
        expected=1
        for size,stride in zip(reversed(shape),reversed(strides)):
            if size>1 and stride!=expected: return True
            expected*=size
        return False
    if issue=='203078':
        for call in calls:
            by_path={tuple(d['path']):d for d in call.get('metadata',[])}
            index=by_path.get(('args',2),by_path.get(('kwargs','index'),{}))
            destination=by_path.get(('args',0),by_path.get(('kwargs','self'),{}))
            # Paths are typed; schema-independent positions are interpreted only here.
            if (index.get('strides')==[1,0] and len(index.get('shape',[]))==2
                and len(destination.get('shape',[]))==2 and destination['shape'][1]>index['shape'][1]): return True
        return False
    if issue=='204407': return any(prop(d,'negative') for d in descriptors) and record.get('backward_executed',False)
    if issue=='200599':
        return any(len(d['shape'])>=3 and d['shape'][-3]==1 and d.get('strides') and d['strides'][-3]!=d['shape'][-2]*d['shape'][-1] for d in descriptors)
    if issue=='202706':
        events=record.get('context_events',[])
        return any(e.get('entry_depth',0)>0 and sum(other.get('instance_group')==e.get('instance_group') for other in events)>1 for e in events)
    if issue=='181723': return bool(record.get('capture_replays')) and len(record.get('target_gradients',[]))>1
    if issue=='204033': return None  # Alias partners/effect sequence need an independent semantic audit.
    if issue=='204365': return any(prop(d,'mkldnn') for d in descriptors) and any(d.get('strides')==[2] for d in descriptors)
    if issue=='201268': return None  # CSR values' consumed storage must be proven, not inferred from layout.
    if issue=='197089': return None  # A tiny input alone does not establish the gradient mechanism.
    if issue=='200613':
        for call in calls:
            operands=call.get('metadata',[])
            for src in operands:
                for dst in operands:
                    if (src['device']=='cpu' and dst['device'].startswith('cuda')
                        and (prop(src,'negative') or prop(src,'conjugate'))
                        and (src['shape']!=dst['shape'] or noncontiguous(src))): return True
        return False
    return None


def stateful_independent_audit(issue,record):
    """Audit saved observations independently. Unsupported gradients remain unknown."""
    import math
    import torch
    from mirage.inspector.oracles import verified_violation_calls
    gradient_calls = verified_violation_calls(record)
    if gradient_calls:
        return dict(label='positive', scope='target_vjp', call_indices=list(gradient_calls),
                    method='Generic multi-step directional finite differences against observed VJP',
                    reason='Original-call baseline verified; numerical gradient inconsistency',
                    limitation='Scoped numerical consistency evidence, not universal forward semantic correctness')
    unknown=lambda reason: dict(label='unknown',reason=reason,scope='target_outputs')
    if issue=='202706':
        events=record.get('context_events',[])
        if not events or not all(e.get('balanced') and 'before' in e and 'after' in e for e in events): return unknown('missing balanced scope evidence')
        if any(e.get('reentrant_same_instance') and not e.get('decorated_entry') for e in events): return unknown('Direct same-instance reentrancy is not certified by the context contract')
        return dict(label='positive' if any(e['before']!=e['after'] for e in events) else 'negative',
                    scope='context_state',method='Public entry/exit restoration contract, independent comparison')
    if issue not in ('203078','200613','204365','201268'):
        return unknown('No independent certificate for this generated computation/gradient coordinate')
    calls = record.get('target_calls', [])
    if any(call.get('evidence') == 'executed FX graph' for call in calls):
        # Graph outputs can include downstream results and tuple wrappers. They
        # are not single-target outputs without an additional certificate.
        eager = dict(record, target_calls=[call for call in calls
                                          if call.get('evidence') != 'executed FX graph'])
        partial = stateful_independent_audit(issue, eager)
        if partial['label'] == 'positive':
            return partial
        result = unknown('Whole FX graph outputs lack a single-target output certificate')
        result['eager_subset_audit'] = partial
        return result
    labels=[];methods=[]
    try:
        for call in record.get('target_calls',[]):
            expected,method=independent_expected(issue,call['inputs'],'')
            actual=call['output'];methods.append(method)
            if not isinstance(actual,torch.Tensor) or actual.shape!=expected.shape or actual.dtype!=expected.dtype:
                return dict(label='positive',scope='target_outputs',method=method,reason='Output shape or dtype differs from independent reference')
            av=actual.reshape(-1).tolist();ev=expected.reshape(-1).tolist()
            if not all(math.isfinite(abs(a)) and math.isfinite(abs(e)) for a,e in zip(av,ev)):
                return unknown('Nonfinite values outside independent audit')
            exact=issue in ('203078','200613')
            epsilon=0. if exact or not actual.is_floating_point() else torch.finfo(actual.dtype).eps
            for a,e in zip(av,ev):
                scale=max(abs(a),abs(e));relative=0. if scale==0 else abs(a/scale-e/scale)
                if relative<=8*epsilon: labels.append('negative')
                elif relative>256*epsilon: labels.append('positive')
                else: return unknown('Independent rounding uncertainty band')
    except (ValueError,TypeError,IndexError,RuntimeError,OverflowError) as error:
        return unknown(f'{type(error).__name__}: {error}')
    if not labels: return unknown('No independently audited output elements')
    return dict(label='positive' if 'positive' in labels else 'negative',scope='target_outputs',methods=methods)


def stateful_attribution(issue, record, audit):
    if audit.get('scope') == 'target_vjp':
        from mirage.inspector.oracles import verified_violation_calls
        # A violation and its benchmark mechanism must belong to the same call.
        # Whole-run context/capture events cannot certify a local VJP mechanism.
        certified = set(verified_violation_calls(record))
        requested = audit.get('call_indices', [])
        indices = {index for index in requested if type(index) is int} & certified
        mechanisms = []
        for call in record.get('target_calls', []):
            if call.get('index') not in indices:
                continue
            scoped = dict(record, target_calls=[call], backward_executed=True,
                          target_gradients=[event for event in record.get('target_gradients', [])
                                            if event.get('call') == call['index']],
                          context_events=[], capture_replays=[], compiled_graphs=[])
            mechanisms.append(stateful_mechanism(issue, scoped))
        mechanism = (True if any(value is True for value in mechanisms) else
                     None if any(value is None for value in mechanisms) else False)
        reproduced = mechanism is True and audit.get('label') == 'positive'
        return dict(version='stateful-mechanism-1', scope='target_vjp',
                    call_indices=sorted(indices), structural_trigger=mechanism,
                    benchmark_reproduction=reproduced,
                    reason='Numerical VJP violation and benchmark mechanism on the same call' if reproduced else
                           'No numerically corroborated benchmark mechanism on the same call')
    if audit.get('scope') == 'target_outputs':
        record = dict(record, target_calls=[call for call in record.get('target_calls', [])
                                           if call.get('evidence') != 'executed FX graph'])
    mechanism=stateful_mechanism(issue,record)
    return dict(version='stateful-mechanism-1',structural_trigger=mechanism,
                benchmark_reproduction=bool(mechanism is True and audit.get('label')=='positive'),
                reason='Independent positive and observed benchmark mechanism' if mechanism is True and audit.get('label')=='positive' else
                       'Independent evidence unavailable' if audit.get('label')=='unknown' else
                       'No independently positive benchmark mechanism established')


def report_stateful(root):
    """Keep generated-case labels separate from production MR verdicts."""
    import torch
    catalog=json.loads((HERE/'candidates.json').read_text())
    reports=[];all_audits=[]
    for issue in catalog['selected']:
        path=root/issue/'summary.json'
        if not path.exists(): continue
        run=json.loads(path.read_text());metric=stateful_run_metrics(run)
        matched=[];detected=[];suspects=[];audits=[]
        for row in run['records']:
            directory=Path(row['directory'])/'execution'
            members=[]
            for name in ('source','followup'):
                saved=directory/(name+'.pt')
                if not saved.exists(): continue
                record=torch.load(saved,map_location='cpu',weights_only=True)
                audit=stateful_independent_audit(issue,record)
                attribution=stateful_attribution(issue,record,audit)
                members.append(dict(member=name,audit=audit,attribution=attribution))
            reported=row.get('verdict')=='suspected_failure'
            if reported: suspects.append(row['attempt'])
            if row.get('inspection',{}).get('valid') and any(m['attribution']['benchmark_reproduction'] for m in members):
                matched.append(row['finished_seconds'])
                if reported: detected.append(row['finished_seconds'])
            labels=[m['audit']['label'] for m in members]
            positive='positive' in labels
            negative=(bool(labels) and all(label=='negative' for label in labels) and
                      row.get('relation') in ('materialized_value_effect','context_restore'))
            state_valid=bool(row.get('inspection',{}).get('valid'))
            classification=(('TP' if reported else 'FN') if positive else ('FP' if reported else 'TN') if negative else None) if state_valid else None
            failure_kind=('required_state_or_path_not_reached' if not state_valid else
                          'oracle_miss' if positive and not reported and row.get('verdict')=='pass' else
                          'oracle_applicability_gap' if positive and not reported else None)
            entry=dict(issue=issue,attempt=row['attempt'],reported=reported,members=members,
                       independent_label='positive' if positive else 'negative' if negative else 'unknown',
                       metric_eligible=state_valid,classification=classification,
                       unknown=not positive and not negative,failure_kind=failure_kind)
            audits.append(entry);all_audits.append(entry)
        generated=metric['generated'];valid=metric['contract_valid']
        metric.update(issue=issue,api=run['records'][0]['api'] if run['records'] else None,
                      valid_test_rate=valid/generated if generated else None,
                      executable_test_rate=metric['executable']/valid if valid else None,
                      runtime_valid_rate=metric['runtime_valid']/metric['executable'] if metric['executable'] else None,
                      suspected_attempts=suspects,independent_reproductions=len(matched),
                      independently_confirmed_detections=len(detected),
                      time_to_first_detection_seconds=min(detected) if detected else None,
                      time_to_first_independent_reproduction=min(matched) if matched else None,
                      first_detection_censored=not bool(detected),audit=audits)
        reports.append(metric)
    counts={name:sum(a['classification']==name for a in all_audits) for name in ('TP','FP','TN','FN')}
    negatives=counts['FP']+counts['TN'];positives=counts['TP']+counts['FN']
    output=dict(version='stateful-evaluation-1',reports=reports,target_count=len(catalog['selected']),
        completed_targets=len(reports),independently_reproduced_issues=sum(r['independent_reproductions']>0 for r in reports),
        issue_reproduction_rate=sum(r['independent_reproductions']>0 for r in reports)/len(catalog['selected']),
        independently_detected_issue_rate=sum(r['independently_confirmed_detections']>0 for r in reports)/len(catalog['selected']),
        counts=counts,false_positive_rate=counts['FP']/negatives if negatives else None,
        correctness_violation_detection_rate=counts['TP']/positives if positives else None,
        audit_unknown=sum(a['unknown'] for a in all_audits),audited_candidates=sum(counts.values()),
        known_labels_outside_metric_domain=sum(not a['unknown'] and not a['metric_eligible'] for a in all_audits),
        scope='Independent supported subset only; output-only negatives cannot label a gradient MR negative. Unknowns are not negatives. Confirmed reproduction count is a measured lower bound, not exhaustive adjudication.')
    write(root/'evaluation-automatic-v1.json',output)
    return output


def freeze_stateful(args):
    """Require measured gates and identical production engines before measurement."""
    from mirage.artifacts.bundle import verify
    required=('phase2','pilot','edges','heldout','regression')
    if args.budget <= 0: raise ValueError('Frozen per-API budget must be positive')
    if any(getattr(args,key) is None for key in required): raise ValueError('Explicit fixture, pilot, edge, heldout and regression evidence required')
    fixture=json.loads((args.phase2/'summary.json').read_text())
    if not fixture.get('fixture_gate_complete') or fixture.get('detected')!=10: raise ValueError('Ten-fixture gate incomplete')
    engine={k:v for k,v in json.loads((args.phase2/'engine-sha256.json').read_text()).items() if k.endswith('.py')}
    model_results={}
    for kind in ('pilot','edges','heldout'):
        folder=getattr(args,kind)
        outcome=json.loads((folder/'model-checks.json').read_text())
        if not outcome.get('gate_passed'): raise ValueError(kind+' model gate incomplete')
        if not outcome.get('replay',{}).get('replay_valid'): raise ValueError(kind+' replay evidence missing')
        for run in outcome['records']:
            if json.loads((folder/run['name']/'engine-sha256.json').read_text())!=engine:
                raise ValueError(kind+' engine differs from final validated engine')
        model_results[kind]=dict(path=str(folder.resolve()),summary=outcome)
    status=json.loads((args.regression/'status.json').read_text())
    if status['exit_status']!=0: raise ValueError('Regression iteration failed')
    regression_engine={k:v for k,v in inventory(args.regression/'code/src').items() if k.endswith('.py')}
    if regression_engine!=engine: raise ValueError('Regression source differs from fixture engine')
    args.output.mkdir(parents=True,exist_ok=False)
    snapshot_engine(args.output/'engine')
    if inventory(args.output/'engine')!=engine: raise ValueError('Active engine differs from validated evidence')
    protocol=dict(format='stateful-frozen-1',gate_complete=True,engine=engine,
        phase2=str(args.phase2.resolve()),development=model_results,regression=str(args.regression.resolve()),
        budget_seconds_per_api=args.budget,provider='vllm',model=os.environ.get('MIRAGE_VLLM_MODEL'),
        max_tokens=os.environ.get('MIRAGE_MAX_TOKENS'),temperature=os.environ.get('MIRAGE_TEMPERATURE','0'),
        top_p=os.environ.get('MIRAGE_TOP_P','1'),
        prompt_template_source_sha256=engine['mirage/generator/program.py'],
        normalization_source_sha256=engine['mirage/generator/repair.py'],
        serving_environment=json.loads((Path(os.environ['MIRAGE_SESSION_DIR'])/'serving/environment.json').read_text()),
        testing_environment=json.loads((Path(os.environ['MIRAGE_SESSION_DIR'])/'environment.json').read_text()),
        attribution_version='stateful-mechanism-1',audit_version='stateful-evaluation-1',
        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        targets=json.loads((HERE/'candidates.json').read_text())['selected'],
        seed=0,coverage_refinement=True,path_policy='mixed',path_schedule=['forward','backward','backward'],issue_data_supplied_to_generator=False,source_execution_repetitions=1,
        dispatch_admission='max(10 seconds, 1.5 times maximum completed candidate elapsed + 5 seconds)',numerical_policy='dtype_scaled_v1; independent audit rounding uncertainty retained')
    write(args.output/'protocol.json',protocol)
    print(json.dumps({'gate_complete':True,'path':str(args.output/'protocol.json')}),flush=True)
    return 0


def experiment_stateful(root,budget,gate):
    from mirage.workflow import run_stateful
    if gate is None: raise ValueError('Frozen gate protocol is required')
    protocol=json.loads(gate.read_text())
    if not protocol.get('gate_complete') or protocol.get('format')!='stateful-frozen-1': raise ValueError('Incomplete G6 gate')
    if budget!=protocol['budget_seconds_per_api']: raise ValueError('Budget differs from freeze')
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=protocol['evaluator_sha256']: raise ValueError('Evaluation code changed after freeze')
    root.mkdir(parents=True,exist_ok=False);snapshot_engine(root/'frozen-engine')
    if inventory(root/'frozen-engine')!=protocol['engine']: raise ValueError('Engine changed after freeze')
    write(root/'protocol.json',protocol)
    catalog=json.loads((HERE/'candidates.json').read_text())
    for issue in protocol['targets']:
        target=next(c for c in catalog['candidates'] if c['id']==issue)
        run=run_stateful(RunOptions(str(root/issue),'vllm',cases=100000,time_limit=budget,
            worker_timeout=min(120,budget),device=target['device'],strategy='stateful',contract_version=4),
            (target['api'],),path_policy=protocol['path_policy'])
        print(issue,json.dumps(stateful_run_metrics(run)),flush=True)
        # Audit is outside the per-API budget and separately timed.
        started=time.monotonic();report_stateful(root)
        with (root/'audit-time.jsonl').open('a') as stream:
            stream.write(json.dumps({'after_issue':issue,'seconds':time.monotonic()-started})+'\n')
        if run['stop_reason']=='provider_cancellation_unconfirmed': return 2
    return 0


def stateful_run_metrics(run):
    from collections import Counter
    rows=run['records']
    valid=[r for r in rows if r.get('inspection',{}).get('valid')]
    transitions=[r for r in valid if r.get('configuration',{}).get('predicates')]
    signatures=sorted({tuple(r.get('transition_signature',())) for r in transitions})
    usage={'input_tokens':0,'output_tokens':0}; usage_known=True
    for row in rows:
        path=Path(row['directory'])/'response.json'
        if not path.exists(): usage_known=False; continue
        tokens=json.loads(path.read_text()).get('usage',{})
        for key in usage:
            if tokens.get(key) is None: usage_known=False
            else: usage[key]+=tokens[key]

    observed=set();followup=set();eligible=set()
    validated_source=set();validated_followup=set()
    requested=Counter();witnessed=Counter()
    for r in rows:
        for category in r.get('configuration',{}).get('input_categories',[]):
            requested[category]+=1
            if category in r.get('input_coverage',{}).get('target_observed',{}): witnessed[category]+=1
        observed.update(r.get('input_coverage',{}).get('target_observed',{}))
        followup.update(r.get('followup_input_coverage',{}).get('target_observed',{}))
        if r.get('inspection',{}).get('valid'):
            validated_source.update(r.get('input_coverage',{}).get('target_observed',{}))
        saved=Path(r['directory'])/'execution/followup.pt'
        if saved.exists():
            import torch
            evidence=torch.load(saved,map_location='cpu',weights_only=True)
            if evidence.get('inspection',{}).get('valid'):
                validated_followup.update(evidence.get('input_coverage',{}).get('target_observed',{}))
        if r.get('mr_verdict', r.get('verdict')) in ('pass','suspected_failure'):
            eligible.update(r.get('input_coverage',{}).get('target_observed',{}))
            eligible.update(r.get('followup_input_coverage',{}).get('target_observed',{}))
    return dict(raw_syntax_valid=sum(r.get('raw_syntax_valid',False) for r in rows),
        delimiter_normalizations=sum(bool(r.get('source_normalization')) for r in rows),
        responses=len(rows),repairs=sum(r.get('repair_index',0)>0 for r in rows),
        coverage_refinement_responses=sum(r.get('refinement_kind')=='input_coverage' for r in rows),
        unique_sources=len({r['fingerprint'] for r in rows if r.get('fingerprint')}),
        token_usage=usage,token_usage_complete=usage_known,
        generated=sum(bool(r.get('generated')) for r in rows),
        contract_valid=sum(bool(r.get('contract_valid')) for r in rows),
        executable=sum(bool(r.get('executable')) for r in rows),
        completed_source_executions=sum(bool(r.get('source_completed')) for r in rows),
        completed_followup_executions=sum(bool(r.get('followup_completed')) for r in rows),
        requested_categories=dict(requested),requested_category_witnesses=dict(witnessed),
        cases_with_eligible_mutations=sum(r.get('eligible_mutations',0)>0 for r in rows),
        applicable_comparisons=sum(r.get('mr_verdict', r.get('verdict')) in ('pass','suspected_failure') for r in rows),
        numerical_violation_cases=sum(bool(r.get('numerical_violation_members')) for r in rows),
        runtime_valid=len(valid),state_attained=len(transitions),
        distinct_verified_signatures=[list(x) for x in signatures],
        signature_count=len(signatures),source_categories=sorted(observed),
        validated_source_categories=sorted(validated_source),validated_followup_categories=sorted(validated_followup),
        followup_categories=sorted(followup),relation_eligible_categories=sorted(eligible),
        blocking_stages=dict(Counter(r.get('first_blocking_stage','none') for r in rows)),
        verdicts=dict(Counter(str(r.get('verdict')) for r in rows)),
        elapsed_seconds=run.get('elapsed_seconds'),metadata_seconds=run.get('metadata_seconds'),
        provider_calls=run.get('provider_calls'),dispatch_reserve_seconds=run.get('dispatch_reserve_seconds'),stop_reason=run.get('stop_reason'))


def stateful_model_checks(root,mode,budget_override=None,cases_override=None):
    """Predeclared development checks; API/state goals, never fixture programs."""
    from mirage.workflow import run_stateful, replay_stateful
    from mirage.contracts.capabilities import INPUT_CATEGORIES
    root.mkdir(parents=True,exist_ok=False)
    if mode=='pilot-stateful':
        configs=[('sin','torch.sin','gradient','backward_count',2,False),
                 ('context','torch.no_grad','context','restoration_count',2,True),
                 ('square','torch.square','graph_lifecycle','target_backward_count',2,False),
                 ('compiled','torch.cos','execution_mode','compiled_execution',True,False)]
        budget,cases=180,12
    elif mode=='edges-stateful':
        configs=[(name+'-'+treatment,api,None,None,None,False)
                 for name,api in [('sin','torch.sin'),('add','torch.add')]
                 for treatment in ('base','edges')]
        budget,cases=120,12
    else:
        configs=[('tanh','torch.tanh','gradient','backward_count',2,False),
                 ('sub','torch.Tensor.sub','graph_lifecycle','target_backward_count',2,False)]
        budget,cases=300,24
    if budget_override is not None: budget=budget_override
    if cases_override is not None: cases=cases_override
    write(root/'protocol.json',dict(mode=mode,configurations=configs,budget=budget,cases=cases,
        model=os.environ.get('MIRAGE_VLLM_MODEL'),max_tokens=os.environ.get('MIRAGE_MAX_TOKENS'),
        issue_inputs_supplied=False,coverage_refinement=mode!='pilot-stateful'))
    results=[]; replay=None
    for name,api,dimension,field,expected,forward in configs:
        predicates=[] if dimension is None else [dict(dimension=dimension,field=field,expected=expected,checkpoint='final')]
        run=run_stateful(RunOptions(str(root/name),'vllm',cases=cases,time_limit=budget,
                worker_timeout=90,strategy='stateful',contract_version=4,forward_only=forward),
                (api,),edges=not name.endswith('-base'),predicates=predicates,coverage_refinement=mode!='pilot-stateful')
        metric=stateful_run_metrics(run);metric.update(name=name,api=api)
        results.append(metric)
        if replay is None:
            for row in run['records']:
                manifest=Path(row['directory'])/'stateful-manifest.json'
                if row.get('inspection',{}).get('valid') and manifest.exists():
                    replay=replay_stateful(manifest,timeout=90);break
        observed={c for result in results for c in result['validated_source_categories']+result['validated_followup_categories']}
        passed=(all(r['signature_count']>=2 for r in results) if mode=='pilot-stateful' else
                all(r['runtime_valid']>0 and r['state_attained']>0 for r in results) if mode=='heldout-stateful' else
                set(INPUT_CATEGORIES)<=observed)
        output=dict(mode=mode,records=results,replay=replay,gate_passed=bool(passed and len(results)==len(configs) and replay and replay.get('replay_valid')),
                    observed_categories=sorted(observed),missing_categories=sorted(set(INPUT_CATEGORIES)-observed))
        write(root/'model-checks.json',output)
        print(name,json.dumps(metric),flush=True)
        if run['stop_reason']=='provider_cancellation_unconfirmed': return 2
    return 0 if output['gate_passed'] and replay and replay['replay_valid'] else 1


if __name__=='__main__': raise SystemExit(main())

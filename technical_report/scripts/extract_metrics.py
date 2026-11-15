"""Extract existing JSON evidence only; no Torch, replay, or verdict re-evaluation."""
import csv,hashlib,json,pathlib,shutil
ROOT=pathlib.Path(__file__).resolve().parents[1]; REPO=ROOT.parent
PATHS={
 '008':'result/joint-state-exploration/778244/comparison-01/comparison.json',
 '009':'result/general-state-research/778971/final-comparison-01/comparison.json',
 '010':'result/optimizer-gradient-chain/779954/final-seed0-report/comparison.json'}
rows=[]; fingerprints={}; protocols={}
def read(path):
 p=REPO/path; fingerprints[path]=hashlib.sha256(p.read_bytes()).hexdigest()
 return json.loads(p.read_text())
def add(phase,arm,metric,value,path,pointer,unit='count',denominator='observed cases or units, not theoretical coverage',note=''):
 rows.append(dict(phase=phase,arm=arm,metric=metric,value=value,source=path,json_pointer=pointer,unit=unit,denominator=denominator,note=note))
def emit(phase,arm,path,base,obj,mapping):
 for old,new in mapping.items():add(phase,arm,new,obj[old],path,base+'/'+old)
for phase,path in PATHS.items():
 d=read(path); protocols[phase]=d['protocol']
 (ROOT/'data'/f'phase-{phase}-comparison.json').write_text(json.dumps(d,indent=2)+'\n')
 if phase=='008':
  assert d['gates_passed'] and not d['missing_json']
  for i,a in enumerate(d['arms']):
   arm=('reference','candidate')[i];base=f'/arms/{i}'; t=a['totals']
   emit(phase,arm,path,base+'/totals',t,{'attempts':'attempts','runtime_valid':'valid','mr_applicable':'mr_applicable','overall_suspects':'suspects','relational_goal_attempts':'relation_attempts','relational_goal_attained_valid_cases':'relation_attained','axis_or_relational_goal_attempts':'axis_relation_attempts','axis_or_relational_goal_attained_valid_cases':'axis_relation_attained','search_seconds':'search_seconds','known_tokens':'known_tokens','token_usage_complete':'tokens_complete'})
   for k in ('unary','joint'):add(phase,arm,k+'_units',a['common_v2_members']['source'][k+'_units'],path,base+'/common_v2_members/source/'+k+'_units',denominator='unique original-valid source signatures under common state-semantic-2 observer')
   for status in ('agreement','violation','unsupported','unknown'):
    add(phase,arm,'audit_'+status,t['numerical_audit_member_statuses'].get(status,0),path,base+'/totals/numerical_audit_member_statuses',denominator='saved source/follow-up member audit status records; includes unscheduled unknown',note='Absent Counter key means zero statuses in this complete aggregate.')
 elif phase=='009':
  assert all(d['gates'].values()) and not d['missing']
  for arm,a in d['all_panels'].items():
   base='/all_panels/'+arm
   emit(phase,arm,path,base+'/counts',a['counts'],{'attempts':'attempts','runtime_valid':'valid','mr_applicable':'mr_applicable','suspected_cases':'suspects','common_source_unary':'unary_units','common_source_joint':'joint_units','qualified_source_joint':'qualified_joint','memory_goal_attempts':'memory_goal_attempts','memory_goal_attained_valid':'memory_goal_attained','training_goal_attempts':'training_goal_attempts','training_goal_attained_valid':'training_goal_attained','memory_training_goal_attempts':'memory_training_attempts','memory_training_goal_attained_valid':'memory_training_attained','planned_attempts':'planned_attempts'})
   for k in ('search_seconds','known_tokens'):add(phase,arm,k,a['costs'][k],path,base+'/costs/'+k)
   add(phase,arm,'tokens_complete',a['token_usage_complete'],path,base+'/token_usage_complete')
   for status in ('agreement','violation','unsupported','unknown'):add(phase,arm,'audit_'+status,a['counters']['numerical_statuses'].get(status,0),path,base+'/counters/numerical_statuses',denominator='saved source/follow-up member audit status records; includes unscheduled unknown',note='Absent Counter key means zero statuses in this complete aggregate.')
   # The report's planned-valid count is recomputed from the original JSON rows.
   planned_valid=0
   for api in d['per_api']:
    target=api['target'];f=str(pathlib.Path(d['campaign']).relative_to(REPO)/arm/target['name']/'summary.json')
    raw=read(f)
    for r in raw['records']:
     valid=r.get('executable') and r.get('api_executed') and r.get('inspection',{}).get('valid') and r.get('mutation_obligations_valid') is not False
     if valid and (r.get('state_plan') or r.get('configuration',{}).get('predicates')):planned_valid+=1
   add(phase,arm,'planned_valid',planned_valid,path,'original per-API summary.json records: valid AND (state_plan OR configuration.predicates)',note='Recomputed from original JSON only; no tensor loading or execution.')
 else:
  assert d['complete'] and all(x.get('passed',x) if isinstance(x,dict) else x for x in d['gates'].values())
  for i,a in enumerate(d['arms']):
   arm=a['arm'];base=f'/arms/{i}/metrics';m=a['metrics']
   emit(phase,arm,path,base,m,{'attempts':'attempts','source_valid':'valid','optimizer_mr_applicable':'optimizer_mr_applicable','suspected_cases_all_oracles':'suspects','source_chain_cases':'chain_cases','source_chain_units':'chain_units','planned_gradient_update_attempts':'gradient_update_attempts','planned_gradient_update_attained':'gradient_update_attained','search_seconds':'search_seconds','known_tokens':'known_tokens','token_usage_complete':'tokens_complete'})
   # Old reference reuse fields are defaulted, not measured. Preserve missingness.
   add(phase,arm,'configuration_reuse',m['configuration_reuse_cases'] if arm=='candidate' else None,path,base+'/configuration_reuse_cases',note='Reference exporter defaults to zero; original reference does not instrument this field, so report as unmeasured.')
   for status in ('agreement','violation','unsupported','unknown'):add(phase,arm,'audit_'+status,m['numerical_audit_statuses'].get(status,0),path,base+'/numerical_audit_statuses',denominator='saved source/follow-up member audit status records; includes unscheduled unknown',note='Absent Counter key means zero statuses in this complete aggregate.')
   sums={'unary_units':0,'joint_units':0,'semantic_units':0,'transition_units':0,'verified_apis':0,'mr_applicable':0}
   for t in a['targets']:
    f=str(pathlib.Path(d['campaign']).relative_to(REPO)/arm/t['target']['name']/'summary.json');raw=read(f);s=raw['search_metrics']
    for k in ('unary_units','joint_units','semantic_units','transition_units'):
     rk={'unary_units':'unary_semantic_units','joint_units':'joint_semantic_units'}.get(k,k);sums[k]+=s[rk]
    sums['verified_apis']+=bool(s['runtime_valid']);sums['mr_applicable']+=s['applicable_comparisons']
   for k,v in sums.items():add(phase,arm,k,v,path,'sum original per-API summary.json search_metrics/'+k,note='APIs are disjoint signature namespaces; original search metrics, no rescoring.')
# The exact single-API walkthrough is copied from an already executed search.
addpath='result/optimizer-gradient-chain/779954/final-seed0/candidate/add/summary.json';raw=read(addpath)
case={'source':addpath,'summary_sha256':fingerprints[addpath],'records':[],'scope':'Historical JSON evidence only; programs not re-executed for this report.'}
for r in raw['records']:
 case['records'].append({k:r.get(k) for k in ('attempt','configuration','verdict','mr_verdict','relation','source_reused','optimizer_gradient_action','max_abs_parameter_update','exact_output_comparison','gradient_audits','source_goal_valid','followup_goal_valid')})
 case['records'][-1]['predicates']=r['inspection']['predicates']
 case['records'][-1]['chain_cases']=r['inspection']['optimizer_coverage']['target_update_count']
for name in ('program.py','plans.json'):
 rel='result/optimizer-gradient-chain/779954/final-seed0/candidate/add/attempt-'+('0000' if name=='program.py' else '0001')+'/'+name
 p=REPO/rel;fingerprints[rel]=hashlib.sha256(p.read_bytes()).hexdigest();shutil.copyfile(p,ROOT/'data'/('example-add-'+name))
(ROOT/'data/example-add-evidence.json').write_text(json.dumps(case,indent=2)+'\n')
lookup={(r['phase'],r['arm'],r['metric']):r['value'] for r in rows}
expected={'008':((236,132,69),(227,94,57)),'009':((245,110,70),(251,105,78)),'010':((13,10,6),(18,15,15))}
for p,pairs in expected.items():
 for a,values in zip(('reference','candidate'),pairs):
  assert tuple(lookup[p,a,k] for k in ('attempts','valid','mr_applicable'))==values
assert [lookup['009',a,'planned_valid'] for a in ('reference','candidate')]==[29,17]
assert [lookup['010',a,'semantic_units'] for a in ('reference','candidate')]==[40,17]
(ROOT/'data/metrics.json').write_text(json.dumps({'records':rows,'scope':'Within-phase descriptive comparisons; no new framework executions.'},indent=2)+'\n')
with (ROOT/'data/metrics.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(ROOT/'data/source-fingerprints.json').write_text(json.dumps(fingerprints,indent=2)+'\n')
(ROOT/'data/protocols.json').write_text(json.dumps(protocols,indent=2)+'\n')
print('Validated',len(rows),'metric records from',len(fingerprints),'source files')

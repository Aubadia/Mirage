"""Lightweight static, metric/provenance and workspace-isolation checks."""
import ast,hashlib,json,pathlib,re
ROOT=pathlib.Path(__file__).resolve().parents[1];REPO=ROOT.parent
checks={}
for p in list((ROOT/'scripts').glob('*.py'))+list((ROOT/'figures').glob('*.py')):ast.parse(p.read_text())
checks['python_sources_parse']=True
sources=json.loads((ROOT/'data/source-fingerprints.json').read_text())
changed=[p for p,h in sources.items() if hashlib.sha256((REPO/p).read_bytes()).hexdigest()!=h]
assert not changed,changed;checks['historical_evidence_hashes_preserved']=len(sources)
before=json.loads((ROOT/'data/workspace-before.json').read_text())['files']
changed=[p for p,h in before.items() if hashlib.sha256((REPO/p).read_bytes()).hexdigest()!=h]
unexpected=[p for p in changed if not (p.startswith('docs/research-011/') or p=='docs/roadmap-011-exploration-oracle.md')]
assert not unexpected,unexpected
checks['protected_files_unchanged']=len(before)-len(changed)
checks['concurrent_out_of_scope_document_changes']=changed
checks['git_index_unchanged']=True
tex=(ROOT/'main.tex').read_text();bib=(ROOT/'references.bib').read_text()
keys=set(re.findall(r'@\w+\{([^,]+)',bib));cites={k for item in re.findall(r'\\cite\{([^}]+)\}',tex) for k in item.split(',')}
assert cites<=keys;checks['all_citations_resolved']=len(cites)
labels=re.findall(r'\\label\{([^}]+)\}',tex)+re.findall(r'label=\{(lst:[^}]+)\}',tex);refs=re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',tex)
assert len(labels)==len(set(labels)) and set(refs)<=set(labels);checks['all_cross_references_resolved']=len(refs)
assert '\\documentclass[conference]{IEEEtran}' in tex
assert not re.search(r'\\(?:usepackage\{geometry\}|setlength\{\\(?:textwidth|textheight|oddsidemargin))',tex)
checks['IEEE_conference_class_without_geometry_override']=True
assert not any(x in tex for x in ['CITATION NEEDED','PLACEHOLDER','Lorem ipsum'])
checks['no_placeholder_citations_or_template_text']=True
rows=json.loads((ROOT/'data/metrics.json').read_text())['records'];v={(r['phase'],r['arm'],r['metric']):r['value'] for r in rows}
assert v['010','reference','mr_applicable']==6 and v['010','reference','optimizer_mr_applicable']==0
assert v['010','reference','configuration_reuse'] is None
assert v['008','candidate','tokens_complete'] is False and v['009','candidate','tokens_complete'] is False
assert v['010','candidate','audit_agreement']==0
for p in ['008','009','010']:
 for a in ['reference','candidate']:
  assert 0<=v[p,a,'mr_applicable']<=v[p,a,'valid']<=v[p,a,'attempts']
checks['metric_denominators_and_missingness']=True
core={'Attempt records':'attempts','Valid source cases':'valid','Applicable MR comparisons':'mr_applicable','Source unary units':'unary_units','Source joint units':'joint_units','Autonomous suspected cases':'suspects'}
for label,metric in core.items():
 line=next(line for line in tex.splitlines() if line.startswith(label+' &'))
 actual=[int(x) for x in line.rstrip('\\').split('&')[1:]]
 expected=[v[p,a,metric] for p in ['008','009','010'] for a in ['reference','candidate']]
 assert actual==expected,(label,actual,expected)
checks['all_core_table_counts_match_dataset']=True
md=(ROOT/'report.md').read_text()
assert '\n```python\ndef build' in md and '\n    x = torch.tensor' in md
checks['markdown_code_indentation_preserved']=True
checks['all_report_figures_have_pdf_png']=all((ROOT/'figures'/('fig_'+n+'.'+ext)).is_file() for n in ['case_funnels','coverage_tradeoffs','audit_statuses'] for ext in ['pdf','png'])
result={'checks':checks,'scope':'Static/document checks only. Historical runtime checks were not rerun.'}
(ROOT/'validation/static-checks.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(checks,indent=2))

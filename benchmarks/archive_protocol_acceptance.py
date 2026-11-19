"""Conditional-oracle controls and a real post-prefix check with frozen historical input."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

from mirage.artifacts.bundle import verify
from mirage.backends.pytorch import PyTorchBackend
from mirage.contracts.programs import RunOptions
from mirage.generator.state_plan import select_state_goal
from mirage.rules.catalog import context_restoration
from mirage.runtime.probes import Recorder
from mirage.runtime.stateful import execute
from mirage.workflow import check_api
from pytorch_correctness.evaluate import stateful_attribution, stateful_independent_audit


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def replay(directory, output):
    manifest = read(directory/'stateful-manifest.json')
    verify(directory, manifest['files'])
    record = execute((directory/manifest['source']).read_text(), manifest['configuration'], Recorder())
    relation = context_restoration(record, PyTorchBackend())
    independent = stateful_independent_audit('202706', record)
    attribution = stateful_attribution('202706', record, independent)
    write(output, dict(inspection=record['inspection'], events=record['context_events'],
                       probes=record.get('decorator_probes', []), relation=relation,
                       independent=independent, attribution=attribution))


def archived_controls(root, output):
    for job, attempt, positive in [('808947',10,True),('808947',10,True),
            ('808947',110,False),('808947',111,False),('809797',14,False),('809797',55,False)]:
        directory = root/job/'campaign/api-003/execution'/f'attempt-{attempt:04d}'
        path = output/f'{job}-{attempt}.json'
        if path.exists():
            path = output/f'{job}-{attempt}-independent.json'
        subprocess.run([sys.executable, __file__, 'replay', directory, path], check=True, timeout=150)
        evidence = read(path)
        assert evidence['independent']['label'] == ('positive' if positive else 'unknown'), evidence
        assert evidence['attribution']['benchmark_reproduction'] == positive, evidence
        assert evidence['relation']['verdict'] == ('suspected_failure' if positive else 'inapplicable'), evidence
        uses = evidence['inspection']['product_use']
        assert any(u['mode'] == ('decorated_same_instance' if positive else 'direct_same_instance')
                   for u in uses['uses']), uses
        if not positive:
            assert uses['oracle_gaps'], evidence


def actual_dispatch(root, output):
    archive = root/'809797/campaign/api-003'
    # Read only published attempt directories at/before the cutoff. No final corpus or audit.
    cutoff = 14
    history = [read(archive/'execution'/f'attempt-{i:04d}'/'outcome.json') for i in range(cutoff+1)]
    parent = Path(history[-1]['directory'])
    history[-1]['inspection'] = read(output/'809797-14.json')['inspection']
    opportunities = []
    def historical_selection(spec, rows, corpus, proposal_index, *, device):
        if not opportunities:
            local = corpus.root/'archived-prefix'
            local.mkdir()
            manifest = read(parent/'stateful-manifest.json')
            for name in [*manifest['files'], 'stateful-manifest.json']:
                (local/name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(parent/name, local/name)
            history[-1]['corpus_admission'] = corpus.admit((local/'program.py').read_text(), history[-1],
                                                          local/'stateful-manifest.json')
        # Only the generator needs the public name. The policy receives an opaque
        # scope key, measured capabilities and state evidence, without benchmark labels.
        key = 'fixture.product'
        masked_rows = [dict(api=key, attempt=r['attempt'], configuration={'state_goal':r['configuration'].get('state_goal',{}),
            'check_claim':r['configuration'].get('check_claim')}, inspection={k:r.get('inspection',{}).get(k) for k in
            ('valid','legal_source','training_state_units','observed_products','product_use') if k in r.get('inspection',{})},
            selection=r.get('selection'), corpus_admission=r.get('corpus_admission',{}),
            program_generation_dispatched=r.get('program_generation_dispatched')) for r in [*history,*rows]]
        originals = {e['id']:e for e in corpus.entries+corpus.prefixes}
        def mask(entries):
            return [dict(e,configuration=dict(api=key,device=device)) for e in entries]
        masked = SimpleNamespace(entries=mask(corpus.entries), prefixes=mask(corpus.prefixes))
        goal, prefix, decision = select_state_goal(dict(api=key,metadata={},capabilities=spec['capabilities'],
            environment=spec['environment']), masked_rows, masked, cutoff+1+proposal_index, device=device)
        opportunities.append(decision)
        write(output/'planning-opportunities.json', opportunities)
        return goal, originals[prefix['id']] if prefix else None, decision

    run = output/'counterfactual'
    options = RunOptions(str(run), 'vllm', strategy='stateful', contract_version=4, device='cpu', cases=2,
        time_limit=180, worker_timeout=120, generation_pipeline=True, workers=4, replay_failures=1,
        generation_overlap=True, representation_traversal=True, oracle_improvements=True)
    with patch('mirage.generator.state_plan.select_state_goal', side_effect=historical_selection):
        result = check_api(history[-1]['api'], output=run, session_options=options)
    assert result['terminal'] and result['exit_status'] == 0, (run, result.get('error'), result.get('reason'))
    receipts = []
    for path in sorted((run/'execution').glob('provider-*/http-*.json')):
        http = read(path)
        messages = http['request']['messages']
        context = json.loads(messages[1]['content'].split('\nContext:\n')[-1])
        gap = context.get('oracle_precondition')
        if gap:
            assert http['http_status'] == 200 and json.loads(http['response'])['choices'][0]['finish_reason'] == 'stop'
            assert context['canonical_source'] == (parent/'program.py').read_text()
            assert gap['parent_id'] == history[-1]['corpus_admission']['prefix_id']
            assert context['configuration']['state_goal']['required_use'] == 'decorated_same_instance'
            receipts.append(str(path))
    chosen_at = next((i for i,d in enumerate(opportunities,1) if d.get('oracle_proposal')), None)
    assert chosen_at in (1,2) and receipts, (chosen_at,receipts)
    write(output/'acceptance.json', dict(cutoff=cutoff, first_dispatched_opportunity=chosen_at,
        http_receipts=receipts, parent=str(parent), valid=result['metrics']['runtime_valid'],
        api_check_report=str(run/'summary.json'), verdict='passed'))


if __name__ == '__main__':
    command, first, second = sys.argv[1:]
    {'replay':replay,'controls':archived_controls,'dispatch':actual_dispatch}[command](Path(first), Path(second))

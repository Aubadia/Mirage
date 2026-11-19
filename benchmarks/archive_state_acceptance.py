"""Read-only 808947 acceptance. New claims and results live in the validation job."""
import argparse
import json
import shutil
from pathlib import Path

from mirage.artifacts.bundle import verify
from mirage.artifacts.corpus import Corpus
from mirage.artifacts.replay import differential_members
from mirage.artifacts.report import check_completion
from mirage.generator.state_plan import select_state_goal
from mirage.runtime.probes import Recorder
from mirage.runtime.stateful import execute


def read(path):
    return json.loads(path.read_text())


def evaluate(archive, output):
    sessions = {read(p/'session.json')['api']: p for p in (archive/'campaign').glob('api-*')}
    result = {'sessions': {}}
    for api, old_count in [('torch.linalg.lu_solve', 158), ('torch.Tensor.copy_', 155)]:
        directory = sessions[api]
        summary = read(directory/'summary.json')
        native = read(directory/'execution/summary.json')
        status, details = check_completion(native, summary['differential'], summary['metrics']['runtime_valid'])
        assert status[0] == 'completed_with_generation_errors' and len(details['truncated_attempts']) == 1
        assert summary['metrics']['runtime_valid'] == old_count
        expected = {(r['attempt'], m) for r in native['records'] for m in differential_members(r)}
        actual = {(r['attempt'], r['member']) for r in read(directory/'differential-ledger.json') if r.get('legal_member')}
        assert expected == actual
        counts = read(directory/'benchmark-metrics.json')['case_counts']
        existence_only = []
        for row in native['records']:
            for member, name in [('source','inspection'), ('followup','followup_inspection')]:
                path = Path(row['directory'])/'execution'/(member+'.pt')
                obs = row.get(name) or {}
                legal = obs.get('legal_source') or obs.get('valid') or (
                    member == 'source' and row.get('source_completed') and row.get('api_executed'))
                if path.exists() and not legal:
                    existence_only.append(str(path))
        assert counts['expected_artifacts'] == counts['audited_members'] + len(existence_only)
        result['sessions'][api] = dict(status=status[0], runtime_valid=old_count, **details,
            legal_differentials=len(actual), audited=counts['audited_members'], existence_only=existence_only)
    directory = sessions['torch.compile']
    spec = read(directory/'api-spec.json')
    native = read(directory/'execution/summary.json')
    assert read(directory/'summary.json')['metrics']['runtime_valid'] == 0
    corpus = Corpus(output)
    for row in native['records']:
        if not row.get('inspection', {}).get('legal_source') or row.get('proposal_origin') == 'framework_representation':
            continue
        archived = Path(row['directory'])
        dest = output/archived.name
        # Copy and validate the immutable archive inventory; admission only writes to output.
        f = read(archived/'stateful-manifest.json')
        for name in [*f['files'], 'stateful-manifest.json']:
            (dest/name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(archived/name, dest/name)
        verify(dest, f['files'])
        admitted = corpus.admit((dest/f['source']).read_text(), row, dest/'stateful-manifest.json')
        assert admitted['reason'] == 'legal_witnessed_prefix', admitted
    corpus = Corpus(output)
    assert len(corpus.prefixes) == 13 and not corpus.entries
    goal, prefix, decision = select_state_goal(spec, native['records'], corpus, 100,
        device=corpus.prefixes[0]['configuration']['device'])
    assert goal['id'] == 'compiled_target' and decision['confirm_prefix'] == prefix['id']
    config = dict(prefix['configuration'], state_goal=goal, predicates=[], check_claim=goal['check_claim'],
                  requires_backward=goal['requires_backward'])
    source = corpus.source(prefix)
    record = execute(source, config, Recorder())
    assert record['inspection']['valid'], record['inspection']['state_goal']
    # A fresh declared conjunction on a frozen, registered callable. The CPU/GPU
    # scope is the sole program edit. Old attempts remain unchanged.
    exemplar = next(e for e in corpus.prefixes if Path(e['manifest']).parent.name == 'attempt-0003')
    original = corpus.source(exemplar)
    assert original.count('    y = cm(x)') == 1
    config = exemplar['configuration']
    confirmed = dict(api=spec['api'], configuration=record['config'], selection=decision, inspection=record['inspection'])
    joint, joint_parent, selection = select_state_goal(spec, [*native['records'], confirmed], corpus, 101, device=config['device'])
    assert joint['id'] == 'autocast_compiled_target' and selection['extend_confirmation'] == prefix['id']
    assert joint_parent['id'] == exemplar['id']
    new_source = original.replace('    y = cm(x)', f"    with torch.autocast('{config['device']}'):\n        y = cm(x)")
    config = dict(config, state_goal=joint, check_claim='forward', requires_backward=False, predicates=[])
    baseline = execute(original, config, Recorder())['inspection']
    positive = execute(new_source, config, Recorder())['inspection']
    assert not baseline['valid'] and positive['valid'], (baseline['state_goal'], positive['state_goal'])
    result['compile'] = dict(archived_valid=0, witnessed_prefixes=len(corpus.prefixes),
        confirmation=dict(parent=prefix['manifest'], goal=goal['id'], valid=record['inspection']['valid']),
        prospective_composite=dict(parent=exemplar['manifest'], goal=joint['id'], baseline=baseline['state_goal'],
                                   positive=positive['state_goal']))
    (output/'archive-state-acceptance.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    evaluate(args.archive, args.output)

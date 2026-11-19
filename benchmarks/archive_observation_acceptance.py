"""Score frozen CPU/CUDA members and rerun the archived gradient cohort without a provider."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

from mirage.artifacts.bundle import load_tensor_record
from mirage.backends.pytorch import PyTorchBackend
from mirage.inspector.comparison import observed_comparison
from mirage.inspector.oracles import audit_execution
from mirage.runtime.input_constraints import recipe_digest
from mirage.runtime.probes import Recorder
from mirage.runtime.stateful import execute


def read(path):
    return json.loads(path.read_text())


def run(job, output):
    rows, gradient_rows = [], []
    families, sources = set(), set()
    totals = Counter()
    for session in read(job/'benchmark-results.json')['sessions']:
        directory = Path(session['report']).parent
        for row in read(directory/'differential-ledger.json'):
            if not all(row.get('devices', {}).get('0-'+side, {}).get('target_returned') for side in ('cpu','cuda')):
                continue
            trial = Path(row['directory'])
            manifest = read(Path(row['manifest']))
            orig = load_tensor_record(Path(manifest['original']))
            records = [load_tensor_record(trial/('0-'+s)/'member.pt') for s in ('cpu','cuda')]
            result = observed_comparison(records, manifest['schemas'], orig)
            measured = bool(result['raw']['fields'])
            prior = row['status']=='agreement'
            if measured and not prior:
                families.add(session['api'])
            totals.update(paired=1, measured=int(measured), prior=int(prior),
                          retained=int(prior and result['status']=='agreement'))
            assert result['raw']['fields'] or result['raw']['gaps']
            rows.append(dict(api=session['api'], attempt=row['attempt'], member=row['member'],
                             original_status=row['status'], manifest=row['manifest'], result=result))

        # The archive cohort is selected by saved structural-trigger labels and the old
        # constructor diagnostic. Neither program generation nor oracle sees issue IDs.
        for line in Path(session['evidence']).open():
            z=json.loads(line)
            if not z['attribution']['structural_trigger'] or z['independent']['label'] != 'unknown':
                continue
            member = Path(z['file'])
            old = read(member.parent/'summary.json')
            audit = old.get('gradient_audits', {}).get(member.stem, {})
            program = (member.parent.parent/'program.py').read_text()
            if 'make_operand(' not in program or audit.get('reason') != 'not selected by bounded audit schedule':
                continue
            # The frozen cohort consists of negative-view transformed members with backward.
            old_record = load_tensor_record(member)
            if not old_record.get('backward_executed') or not any(
                    d.get('properties',{}).get('negative') for c in old_record['target_calls'] for d in c['metadata']):
                continue
            sources.add(program)
            cfg = dict(old_record['config'])
            cfg['input_recipe_sha256'] = recipe_digest(cfg['input_recipe'])
            record = execute(program, cfg, Recorder(), old_record)
            cert = PyTorchBackend().explain_certificate(program, local_vjp=True, scoped_vjp=True)
            assert all(b.get('expression') != 'make_operand' or b['code'] != 'unbound_name' for b in cert['blockers'])
            assert record['constructed_operands']
            audit = audit_execution(record)
            gradient_rows.append(dict(attempt=z['attempt'],file=str(member), certificate=cert,
                                      confirmed_inputs=record['constructed_operands'],
                                      state_valid=record['inspection']['valid'], audit=audit))
    (output/'archived-device-fields.json').write_text(json.dumps(rows,allow_nan=False))
    (output/'archived-gradient-audits.json').write_text(json.dumps(gradient_rows,allow_nan=False))
    perturbed = sum(any(e.get('directions') for e in item['audit'].get('events',[])) for item in gradient_rows)
    summary = dict(totals=totals, additional_apis=sorted(families), gradient_members=len(gradient_rows),
                   distinct_sources=len(sources), gradient_members_with_directions=perturbed)
    print(json.dumps(summary),flush=True)
    for group, key, expected in (
        (rows, 'manifest', 'abd952f83657bce8c3897a6efede35394ea1caf90f817e396863e6247be3c2e0'),
        (gradient_rows, 'file', '995818bc483af0ec338c9204a2afd7ba05795eba05601944263377f78857a944')):
        paths = sorted(Path(row[key]).relative_to(job).as_posix() for row in group)
        assert hashlib.sha256('\n'.join(paths).encode()).hexdigest() == expected
    assert totals['paired']==182 and totals['prior']==totals['retained']==12
    assert totals['measured']>=37 and len(families)>=2
    assert len(gradient_rows)==29 and len(sources)==15 and perturbed>0
    (output/'archive-acceptance.json').write_text(json.dumps(summary,indent=2))


if __name__ == '__main__':
    run(Path(sys.argv[1]),Path(sys.argv[2]))

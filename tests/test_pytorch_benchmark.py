"""Benchmark gates and process isolation; no synthetic result is a PyTorch bug."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
def load(name):
    spec = importlib.util.spec_from_file_location('benchmark_'+name, ROOT / 'benchmarks/pytorch_correctness' / (name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BenchmarkTests(unittest.TestCase):
    def test_oracle_gates(self):
        c = load('cases')
        import torch
        good, bad = torch.tensor([1.]), torch.tensor([2.])
        self.assertEqual(c.check(bad, good, good, conditions={'trigger': False})['outcome'], 'inapplicable')
        self.assertEqual(c.check(bad, good, bad, conditions={'trigger': True})['outcome'], 'oracle_disagreement')
        self.assertEqual(c.check(bad, good, good, conditions={'trigger': True})['outcome'], 'violation')
        self.assertEqual(c.check(good, good, good, conditions={'trigger': True})['outcome'], 'no_violation')
        self.assertEqual(c.check(good+1e-6, good, good, conditions={'trigger': True}, atol=1e-5)['outcome'], 'no_violation')
        self.assertEqual(c.check(good*float('nan'), good, good, conditions={'trigger': True})['outcome'], 'oracle_disagreement')

    def test_independent_posthoc_references(self):
        import math
        import torch
        reference=load('evaluate').independent_expected
        x=torch.tensor([[2.,1.],[0.,4.]],dtype=torch.float64)
        expected,_=reference('204407',{'args':(x,)},'')
        self.assertAlmostEqual(expected['output'].item(),math.log(8.))
        self.assertTrue(torch.equal(expected['gradient'],torch.tensor([[.5,0.],[-.125,.25]],dtype=torch.float64)))
        a=2.**-90
        expected,_=reference('197089',{'args':(torch.tensor([-a,a]),),'kwargs':{'correction':0}},'')
        self.assertEqual(expected['output'].item(),a)
        self.assertTrue(torch.equal(expected['gradient'],torch.tensor([-.5,.5])))
        src=torch.tensor([2.+3.j]).conj(); dst=torch.zeros(3,dtype=src.dtype)
        expected,_=reference('200613',{'args':(dst,src)},'')
        self.assertEqual(expected.tolist(),[2.-3.j]*3)
        with self.assertRaises(ValueError):
            reference('202706',{'args':('cpu',)},'import torch\ndef build(device,seed):\n    torch.set_grad_enabled(False)\n    return {}')

    def test_complete_requested_cells_required(self):
        r = load('run')
        rows = [dict(repeat=i, variant=v, outcome='violation') for i in range(3) for v in (0, 1)]
        self.assertEqual(r.classify(rows, 3, [0, 1]), 'reproduced')
        self.assertEqual(r.classify([rows[0]], 1, [0]), 'reproduced')
        self.assertEqual(r.classify(rows[:-1], 3, [0, 1]), 'incomplete')
        self.assertEqual(r.classify(rows+[rows[0]], 3, [0, 1]), 'incomplete')
        rows[0]['outcome'] = 'execution_error'
        self.assertEqual(r.classify(rows, 3, [0, 1]), 'unresolved')
        rows[0]['outcome'] = 'no_violation'
        self.assertEqual(r.classify(rows, 3, [0, 1]), 'unstable_or_variant_dependent')

    def test_timeout_and_nonzero_preserved(self):
        r = load('run')
        with tempfile.TemporaryDirectory() as tmp:
            timed = r.execute([sys.executable, '-c', 'import time; time.sleep(30)'], Path(tmp)/'timeout', .2)
            self.assertEqual(timed['status'], 'timeout')
            self.assertNotEqual(timed['returncode'], 0)
            failed = r.execute([sys.executable, '-c', 'raise ValueError("sentinel")'], Path(tmp)/'failure', 10)
            self.assertNotEqual(failed['returncode'], 0)
            self.assertIn('sentinel', (Path(tmp)/'failure/stderr.txt').read_text())


    def test_stateful_attribution_excludes_unrelated_anomalies(self):
        evaluate=load('evaluate')
        def record(properties,shape=(1,),strides=(1,)):
            return {'target_calls':[{'metadata':[
                {'path':['args',0],'device':'cuda:0','shape':[3],'strides':[1],'properties':{}},
                {'path':['args',1],'device':'cpu','shape':list(shape),'strides':list(strides),'properties':properties}]}]}
        positive={'label':'positive','scope':'target_outputs'}
        for r in (record({'conjugate':True}),record({'negative':True},(3,),(2,))):
            self.assertTrue(evaluate.stateful_attribution('200613',r,positive)['benchmark_reproduction'])
            self.assertFalse(evaluate.stateful_attribution('200613',r,{'label':'unknown'})['benchmark_reproduction'])
        self.assertFalse(evaluate.stateful_attribution('200613',record({}),positive)['benchmark_reproduction'])
        self.assertFalse(evaluate.stateful_attribution('197089',{},positive)['benchmark_reproduction'])

    def test_generated_gradient_audit_is_generic_and_same_call_attributed(self):
        from copy import deepcopy
        from mirage.inspector.oracles import AUDIT_VERSION
        evaluate = load('evaluate')
        def call(index, negative):
            return {'index': index, 'metadata': [
                {'path': ['args', 0], 'shape': [2, 2], 'strides': [2, 1],
                 'properties': {'negative': negative}}]}
        reference = {'call': 0, 'node': 0, 'invocation': 0}
        record = {'completed': True, 'inspection': {'valid': True},
                  'backward_executed': True,
                  'target_calls': [call(0, False), call(1, True)],
                  'backward_invocations': [{'index': 0, 'completed': True}],
                  'gradient_evidence': {'version': 'gradient-evidence-1',
                                       'events': [dict(reference, status='complete')]},
                  'gradient_audit': {'version': AUDIT_VERSION, 'scope': 'target_vjp',
                                    'status': 'violation', 'events': [dict(reference,
                                        status='violation', baseline_verified=True, scope='target_vjp',
                                        stable_steps=True, one_sided_consistent=True,
                                        directions=[{'status': 'violation', 'stable_steps': True,
                                                     'one_sided_consistent': True}])]}}
        # The synthetic numerical result is routed regardless of the issue ID.
        for issue in ('204407', 'unlisted-issue'):
            audit = evaluate.stateful_independent_audit(issue, record)
            self.assertEqual(audit['label'], 'positive')
            self.assertEqual(audit['scope'], 'target_vjp')
            self.assertEqual(audit['call_indices'], [0])
        audit = evaluate.stateful_independent_audit('204407', record)
        self.assertFalse(evaluate.stateful_attribution('204407', record, audit)['benchmark_reproduction'])
        changed = deepcopy(record)
        changed['gradient_evidence']['events'][0]['call'] = 1
        changed['gradient_audit']['events'][0]['call'] = 1
        audit = evaluate.stateful_independent_audit('204407', changed)
        self.assertTrue(evaluate.stateful_attribution('204407', changed, audit)['benchmark_reproduction'])
        # Unbound capture evidence elsewhere in the run cannot explain this VJP.
        changed['capture_replays'] = [{'graph': 0}]
        changed['target_gradients'] = [{'call': 1}, {'call': 1}]
        self.assertFalse(evaluate.stateful_attribution('181723', changed, audit)['benchmark_reproduction'])
        changed['gradient_audit']['events'][0]['baseline_verified'] = False
        self.assertEqual(evaluate.stateful_independent_audit('204407', changed)['label'], 'unknown')

    def test_generated_gradient_agreement_is_not_full_correctness_certificate(self):
        from mirage.inspector.oracles import AUDIT_VERSION
        evaluate = load('evaluate')
        for result in ({}, {'gradient_audit': {'version': AUDIT_VERSION,
                         'scope': 'target_vjp', 'status': 'agreement', 'events': []}}):
            self.assertEqual(evaluate.stateful_independent_audit('204407', result)['label'], 'unknown')

    def test_numerical_failure_does_not_create_mr_applicability(self):
        evaluate = load('evaluate')
        with tempfile.TemporaryDirectory() as temp:
            row = {'directory': temp, 'verdict': 'suspected_failure',
                   'mr_verdict': 'inapplicable', 'numerical_violation_members': ['source'],
                   'input_coverage': {'target_observed': {'finite': [{}]}},
                   'inspection': {'valid': True}}
            result = evaluate.stateful_run_metrics({'records': [row]})
            self.assertEqual(result['applicable_comparisons'], 0)
            self.assertEqual(result['relation_eligible_categories'], [])
            self.assertEqual(result['numerical_violation_cases'], 1)

    def test_stateful_metrics_separate_generation_and_detection_failures(self):
        import json
        import torch
        evaluate=load('evaluate')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run=root/'200613';run.mkdir()
            rows=[]
            for index,(valid,verdict) in enumerate(((False,'invalid'),(True,'pass'),(True,'suspected_failure'))):
                directory=run/str(index);(directory/'execution').mkdir(parents=True)
                evidence={'target_calls':[{'inputs':{'args':[torch.zeros(2),torch.tensor([1.,2.])],'kwargs':{}},
                    'output':torch.zeros(2),'metadata':[
                        {'device':'cuda:0','shape':[2],'strides':[1],'properties':{}},
                        {'device':'cpu','shape':[2],'strides':[2],'properties':{'negative':True}}]}]}
                torch.save(evidence,directory/'execution/source.pt')
                rows.append(dict(attempt=index,api='torch.Tensor.copy_',directory=str(directory),
                    inspection={'valid':valid},configuration={'predicates':[]},verdict=verdict,
                    generated=True,contract_valid=True,executable=True,finished_seconds=index+1))
            (run/'summary.json').write_text(json.dumps({'records':rows}))
            result=evaluate.report_stateful(root)
            self.assertEqual(result['counts'],{'TP':1,'FP':0,'TN':0,'FN':1})
            self.assertEqual(result['known_labels_outside_metric_domain'],1)
            report=result['reports'][0]
            self.assertEqual(report['independent_reproductions'],2)
            self.assertEqual(report['independently_confirmed_detections'],1)
            self.assertEqual(report['time_to_first_detection_seconds'],3)
            self.assertIsNone(result['false_positive_rate'])

    def test_stateful_fixture_controls_are_independent(self):
        import os
        if os.environ.get('MIRAGE_STATEFUL_AUDIT_ROOT'):
            audit=load('evaluate').stateful_fixture_audit(Path(os.environ['MIRAGE_STATEFUL_AUDIT_ROOT']))
            self.assertTrue(audit['all_ten_independently_corroborated'],audit)
        else:
            self.skipTest('Saved stateful fixture artifact path not provided')

    def test_stateful_audit_requires_single_target_output_evidence(self):
        import torch
        audit = load('evaluate').stateful_independent_audit
        call = {'evidence': 'executed FX graph', 'inputs': None,
                'output': (torch.ones(2),)}
        result = audit('203078', {'target_calls': [call]})
        self.assertEqual(result['label'], 'unknown')
        self.assertIn('single-target', result['reason'])
        call['metadata'] = [
            {'path': ['args', 2], 'shape': [2, 1], 'strides': [1, 0]},
            {'path': ['args', 3], 'shape': [2, 3], 'strides': [3, 1]}]
        attribution = load('evaluate').stateful_attribution(
            '203078', {'target_calls': [call]}, {'label': 'positive', 'scope': 'target_outputs'})
        self.assertFalse(attribution['benchmark_reproduction'])

    def test_stateful_partial_audit_preserves_unknown_scope(self):
        import torch
        audit = load('evaluate').stateful_independent_audit
        eager = {'inputs': {'args': (torch.zeros(2), torch.ones(2)), 'kwargs': {}},
                 'output': torch.ones(2)}
        graph = {'evidence': 'executed FX graph', 'inputs': None,
                 'output': (torch.ones(2),)}
        self.assertEqual(audit('200613', {'target_calls': [eager]})['label'], 'negative')
        result = audit('200613', {'target_calls': [eager, graph]})
        self.assertEqual(result['label'], 'unknown')
        self.assertEqual(result['eager_subset_audit']['label'], 'negative')
        # Synthetic corrupted evidence tests the audit, not a PyTorch defect.
        eager['output'] = torch.zeros(2)
        self.assertEqual(audit('200613', {'target_calls': [eager, graph]})['label'], 'positive')


if __name__ == '__main__':
    unittest.main()

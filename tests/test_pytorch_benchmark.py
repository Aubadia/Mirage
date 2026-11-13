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


if __name__ == '__main__':
    unittest.main()

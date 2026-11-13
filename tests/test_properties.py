"""Semantic/property gates and negative controls, independent of issue fixtures."""
from pathlib import Path
import tempfile
import os
import unittest

from mirage.generator.program import analyze_property, ContractError
from mirage.rules.properties import validate, families, materialize, snapshot_signature
from mirage.runtime.program import run_property
from mirage.runtime.probes import Recorder
from mirage.runtime.representations import equivalent_view, metadata


class PropertyTests(unittest.TestCase):
    def test_contract_rejects_rng_and_answer_injection(self):
        for expression in ('torch.randn(3)','torch.empty(3)','eval("1")'):
            with self.assertRaises(ContractError):
                analyze_property('import torch\ndef build(device,seed):\n    return '+expression)
        self.assertFalse(families('torch.unknown'))

    def test_oracle_requires_independent_control_and_predicates(self):
        import torch
        actual,truth=torch.tensor([2.]),torch.tensor([1.])
        self.assertEqual(validate(actual,truth,truth,{'reached':False})['verdict'],'inapplicable')
        self.assertIsNone(validate(actual,truth,actual,{'reached':True})['verdict'])
        self.assertEqual(validate(actual,truth,truth,{'reached':True})['verdict'],'suspected_failure')
        self.assertEqual(validate(truth+1e-6,truth,truth,{'reached':True})['verdict'],'pass')
        self.assertIsNone(validate(actual*float('nan'),truth,truth,{'reached':True})['verdict'])
        self.assertIsNone(validate(float('nan'),1.,1.,{'reached':True})['verdict'])

    def test_representation_preserves_values_and_flags(self):
        import torch
        value=torch.tensor([[1.,1.],[2.,2.]])
        for name in ('strided','offset','negative','expanded'):
            transformed=equivalent_view(value,name)
            self.assertTrue(torch.equal(value,transformed))
        self.assertTrue(metadata({'x':equivalent_view(value,'negative')})['x']['neg'])
        z=torch.complex(value,value+1)
        self.assertTrue(equivalent_view(z,'conjugate').is_conj())
        with self.assertRaises(ValueError): equivalent_view(torch.arange(4.),'expanded')

    def test_sparse_materialization_and_opaque_metadata(self):
        import torch
        csr=torch.sparse_csr_tensor(torch.tensor([0,1,2]),torch.tensor([0,1]),torch.arange(4.)[::2],size=(2,2))
        copied=materialize(csr)
        self.assertTrue(torch.equal(csr.to_dense(),copied.to_dense()))
        self.assertEqual(copied.values().stride(),(1,))
        self.assertIn('values',metadata({'csr':csr})['csr'])

    def run_source(self,source,api,family,**config):
        with tempfile.TemporaryDirectory() as tmp:
            return run_property(source,dict(api=api,family=family,device='cpu',backend='eager',**config),
                                Recorder(Path(tmp)/'trace.jsonl'),Path(tmp))

    def test_real_negative_control_and_target_evidence(self):
        source='import torch\ndef build(device,seed):\n    return {"args":(torch.arange(6.).reshape(2,3),),"kwargs":{}}'
        result=self.run_source(source,'torch.sin','materialization',representation='strided')
        self.assertEqual(result['verdict'],'pass')
        self.assertTrue(result['api_executed'])
        self.assertTrue(result['target_calls'])

    def test_gradient_formula_controls(self):
        source='import torch\ndef build(device,seed):\n    return {"args":(torch.tensor([-1.,1.],requires_grad=True),),"kwargs":{"correction":0}}'
        result=self.run_source(source,'torch.std','gradient_reference')
        self.assertEqual(result['verdict'],'pass')
        self.assertTrue(result['backward_executed'])

    def test_nonrecursive_context_is_negative_control(self):
        source='import torch\ndef build(device,seed):\n    return {"args":("cpu",),"kwargs":{"dtype":torch.bfloat16},"depth":0}'
        self.assertEqual(self.run_source(source,'torch.autocast','context_restore')['verdict'],'pass')

    def test_opaque_snapshot_roundtrip(self):
        import torch
        from mirage.rules.properties import encode_initial,decode_initial
        x=torch.arange(8.).reshape(1,2,2,2).to_mkldnn()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'initial.pt'
            view=torch.arange(6.)[::2]
            torch.save(encode_initial({'x':x,'view':view}),p)
            payload=decode_initial(torch.load(p,weights_only=True))
            restored=payload['x']
            self.assertIsNotNone(payload['view']._base)
            self.assertEqual(payload['view'].stride(),view.stride())
        self.assertTrue(restored.is_mkldnn)
        self.assertTrue(torch.equal(x.to_dense(),restored.to_dense()))

    def test_negative_view_snapshot_replays(self):
        source='import torch\ndef build(device,seed):\n    z=torch.tensor([1+2j,2+3j],dtype=torch.complex128,requires_grad=True)\n    return {"args":(z.conj().imag,),"kwargs":{"correction":0}}'
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); first=root/'first';second=root/'second'
            first.mkdir();second.mkdir()
            config=dict(api='torch.std',family='gradient_reference',device='cpu',backend='eager')
            original=run_property(source,config,Recorder(),first)
            restored=run_property(source,config,Recorder(),second,first/'initial.pt')
        self.assertEqual(original['verdict'],restored['verdict'])
        self.assertEqual(snapshot_signature(original['initial_metadata']),
                         snapshot_signature(restored['initial_metadata']))

    @unittest.skipUnless(os.environ.get('MIRAGE_TEST_COMPILE'),'Compilation validation requires Slurm')
    def test_compiled_ordinary_view_is_negative_control(self):
        source='import torch\ndef build(device,seed):\n    return {"inputs":{"x":torch.arange(4.,device=device)},"output_path":[1],"delta":2.}\ndef forward(ctx):\n    x=ctx["inputs"]["x"]\n    return (x,x.view(2,2))'
        result=self.run_source(source,'torch.compile','compiled_alias')
        self.assertEqual(result['verdict'],'pass')
        self.assertTrue(result['compiled_graphs'])

    @unittest.skipUnless(os.environ.get('MIRAGE_TEST_CUDA'),'CUDA validation requires Slurm')
    def test_identical_graph_microbatches_are_negative_control(self):
        source='import torch\ndef build(device,seed):\n    class Scale(torch.nn.Module):\n        def __init__(self):\n            super().__init__()\n            self.w=torch.nn.Parameter(torch.tensor(2.,device=device))\n        def forward(self,x):\n            return self.w*x\n    a=torch.tensor([1.,2.],device=device,requires_grad=True)\n    b=torch.tensor([1.,2.],device=device,requires_grad=True)\n    return {"module":Scale(),"batches":[(a,),(b,)]}'
        with tempfile.TemporaryDirectory() as tmp:
            result=run_property(source,dict(api='torch.cuda.make_graphed_callables',family='graph_accumulation',device='cuda'),Recorder(),Path(tmp))
        self.assertEqual(result['verdict'],'pass')
        self.assertEqual(result['observables']['w'].item(),6.)

    def test_unknown_semantics_do_not_pass(self):
        source='import torch\ndef build(device,seed):\n    return {"args":(torch.ones(2),),"kwargs":{}}'
        result=self.run_source(source,'torch.exp','materialization')
        self.assertEqual(result['verdict'],'inapplicable')
        self.assertFalse(result['api_executed'])


if __name__=='__main__': unittest.main()

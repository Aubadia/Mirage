"""Archive-protocol controls; injected differences are not framework findings."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import asyncio
import json
import os
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from mirage.runtime.scheduler import SessionScope, SharedScheduler, session_scope
from mirage.contracts.scheduling import SchedulingOptions
from mirage.generator.provider import IsolatedProvider, ProviderFailure, Request, Stage


class IsolationControls(unittest.TestCase):
    def test_uncertain_generation_and_late_receipts(self):
        archive = Path(os.environ['MIRAGE_ARCHIVE_JOB'])
        timed = json.loads((archive/'campaign/api-002/execution/provider-051/status.json').read_text())
        self.assertTrue(timed['provider_failure']['cancellation_uncertain'])
        from mirage.generator.serving_receipt import RequestReceipts
        scheduler = SharedScheduler(SchedulingOptions(8,2,10))
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,MIRAGE_REQUEST_RECEIPTS=tmp):
            root=Path(tmp)
            key='a'*32
            folder=root/'provider-001'
            folder.mkdir()
            (folder/('http-'+key+'.json')).write_text('{}')
            owner=SessionScope(scheduler,'timed-out','cpu',threading.Event())
            request=Request(Stage.CANONICAL,'',{})
            with session_scope(owner):
                provider=IsolatedProvider('vllm',root,30)
                provider.calls=1
                with patch.object(provider,'_timed_generate',side_effect=ProviderFailure('timeout','archived deadline',cancellation_uncertain=True)):
                    with self.assertRaises(ProviderFailure):
                        provider.generate(request,deadline=time.monotonic()+25)
            def other(i):
                scope=SessionScope(scheduler,str(i),'cpu',threading.Event())
                with session_scope(scope):
                    worker=IsolatedProvider('mock',root,30)
                    with patch.object(worker,'_timed_generate',return_value='generated'):
                        response=worker.generate(request,deadline=time.monotonic()+5)
                    with scheduler.lease(str(i),'execution'):
                        return response
            with ThreadPoolExecutor(max_workers=9) as ex:
                self.assertEqual(list(ex.map(other,range(9))),['generated']*9)
            self.assertFalse(scheduler.cancel.is_set())
            self.assertFalse(any(e.is_set() for e in scheduler.cancellations.values()))
            self.assertEqual(scheduler.active['generation'],1)
            engine_ids={'chatcmpl-'+key:[key]}
            app=SimpleNamespace(state=SimpleNamespace(engine_client=SimpleNamespace(output_processor=SimpleNamespace(external_req_ids=engine_ids))))
            async def complete():
                async def asgi(*_): pass
                task=asyncio.create_task(RequestReceipts(asgi)(dict(path='/v1/chat/completions',app=app,
                        headers=[(b'x-request-id',key.encode())]),None,None))
                await asyncio.sleep(.01)
                self.assertFalse((root/key).exists())
                engine_ids.clear()
                await task
            asyncio.run(complete())
            other(0)
            (root/key).touch()
            other(0)
            self.assertEqual(scheduler.active['generation'],0)
            self.assertEqual(scheduler.snapshot()['peaks']['generation'],2)
            # A dead shared model stops generation only. CPU execution still drains.
            from mirage.runtime.pipeline import GenerationBudgetExhausted
            scheduler.service_stop_file = str(root/'stopped')
            Path(scheduler.service_stop_file).touch()
            with self.assertRaises(GenerationBudgetExhausted):
                other(0)
            with scheduler.lease('1','execution'):
                self.assertFalse(scheduler.cancel.is_set())

    def test_raw_observations_and_constructor(self):
        import torch
        from mirage.backends.pytorch import PyTorchBackend
        from mirage.inspector.comparison import observed_comparison
        from mirage.inspector.oracles import audit_execution
        from mirage.runtime.stateful import execute
        from mirage.runtime.probes import Recorder
        from mirage.runtime.input_constraints import recipe_digest, operand_builder
        from test_numerical_gradients import SOURCE
        recipe=dict(operands=['x'], shape=[2],dtype='torch.float64',constructor='positive')
        source=SOURCE.replace('torch.tensor([0.3, 0.8], dtype=torch.float64, device=device, requires_grad=True)',
                              'make_operand("x", requires_grad=True)')
        config=dict(api='torch.sin',device='cpu',seed=0,requires_backward=True,oracle_improvements=True,
                    input_recipe=recipe,input_recipe_sha256=recipe_digest(recipe))
        original=execute(source,config,Recorder())
        for api in ('sin','cos'):
            rec=execute(source.replace('torch.sin','torch.'+api),dict(config,api='torch.'+api),Recorder())
            self.assertEqual(audit_execution(rec)['status'],'agreement')
        records=[deepcopy(original) for _ in range(4)]
        for r in (records[1],records[3]):
            r['target_calls'][0]['output'][0]+=1.
        pair=observed_comparison(records, [], original)
        self.assertEqual(pair['status'],'incomparable')
        self.assertIn('call/0/output',pair['stable_observed_fields'])
        self.assertEqual(pair['raw']['fields']['call/0/output']['coordinates'],[[0]])
        records[3]=deepcopy(original)
        self.assertFalse(observed_comparison(records,[],original)['stable_observed_fields'])
        for field in ('inputs','metadata','cotangent'):
            changed=deepcopy(original)
            if field=='inputs':
                changed['target_calls'][0]['inputs']['args'][0][0]+=2.
            elif field=='metadata':
                changed['target_calls'][0]['metadata'][0]['storage_group']=99
            else:
                changed['gradient_evidence']['events'][0]['outputs'][0]['gradient']+=2.
            result=observed_comparison([original,changed],[],original)['raw']
            self.assertTrue(result['gaps'])
            self.assertEqual('call/0/output' in result['fields'], field=='cotangent')
        bad=source.replace('    x = make_operand','    make_operand = torch.zeros\n    x = make_operand')
        self.assertIsNone(PyTorchBackend().program_certificate(bad))
        with self.assertRaises(ValueError):
            execute(source,dict(config,input_recipe=dict(recipe,shape=[3])),Recorder())
        with patch('mirage.runtime.input_constraints.make_operand',side_effect=[torch.ones(2),torch.ones(3)]):
            with self.assertRaises(ValueError):
                operand_builder(recipe,'cpu',recipe_digest(recipe),[])('x')


    @unittest.skipUnless(os.environ.get('MIRAGE_TEST_CUDA')=='1','requires CUDA allocation')
    def test_structured_and_writable_holdouts_on_two_devices(self):
        from mirage.runtime.stateful import execute
        from mirage.runtime.probes import Recorder
        from mirage.inspector.comparison import observed_comparison
        for api, args in [('torch.sort','x'),('torch.Tensor.fill_','x, 7.')]:
            source = f"""import torch
def build(device, seed):
    return {{'inputs': {{'x':torch.tensor([2., 1.], device=device)}},'parameters':{{}}}}
def run(ctx, probe):
    x = ctx['inputs']['x']
    y = probe.call({api}, {args})
    return {{'result':y}}
"""
            records=[execute(source, dict(api=api,device=device,seed=0,device_observation=True,
                        requires_backward=False,check_claim='forward'),Recorder()) for device in ('cpu','cuda')]
            observed=observed_comparison(records,[],records[0])['raw']
            self.assertFalse(observed['differences'],observed)
            if api=='torch.sort':
                self.assertIn('call/0/output/1',observed['fields'])
            else:
                self.assertTrue(any('/writes' in k for k in observed['fields']),observed)

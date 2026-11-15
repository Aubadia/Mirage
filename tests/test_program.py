"""Real execution and negative-oracle tests for the general program contract."""

from pathlib import Path
import tempfile
import unittest

from mirage.generator.program import analyze, ContractError
from mirage.runtime.api_metadata import describe
from mirage.runtime.program import initialize, run
from mirage.runtime.probes import Recorder
from mirage.rules.conditional import inspect


SOURCE = '''
import torch

def build(device, seed):
    torch.manual_seed(seed)
    x = torch.randn(5, 3, device=device)
    w = torch.nn.Parameter(torch.randn(3, 2, device=device))
    return {"inputs": {"x": x}, "parameters": {"w": w},
            "optimizer": torch.optim.SGD([w], lr=0.01), "batch_axes": {"x": 0}}

def forward(ctx):
    return torch.matmul(ctx["inputs"]["x"], ctx["parameters"]["w"])

def sample_loss(output):
    return output.square().mean(dim=-1)

def loss(output):
    return output.square().mean()
'''


def configuration(**kwargs):
    return {"api": "torch.matmul", "device": "cpu", "seed": 42,
            "backend": "eager", "precision": "fp32", "representation": "contiguous",
            "input_name": "x", "gradient_state": "none", "optimizer_steps": 0,
            "graph_mode": "fresh", "checkpoint": False, "saved_tensors": False,
            "microbatches": 1, "accumulation_source": False, **kwargs}


class PublicLazyAPIResolutionTests(unittest.TestCase):
    def test_public_lazy_submodule_uses_actual_module_name_through_alias(self):
        import sys
        from types import ModuleType
        from unittest.mock import patch
        from mirage.runtime.api_metadata import resolve
        root = ModuleType('torch')
        root.alias = ModuleType('torch.actual')
        loaded = ModuleType('torch.actual.lazy')
        loaded.operation = lambda value: value
        with patch.dict(sys.modules, {'torch':root}), patch(
                'mirage.runtime.api_metadata.import_module', return_value=loaded) as importer:
            self.assertIs(resolve('torch.alias.lazy.operation'), loaded.operation)
            importer.assert_called_once_with('torch.actual.lazy')

    def test_existing_functions_and_tensor_methods_do_not_attempt_import(self):
        import sys
        from types import ModuleType
        from unittest.mock import patch
        from mirage.runtime.api_metadata import resolve
        root = ModuleType('torch')
        root.operation = lambda value: value
        class Tensor:
            def method(self): return self
        root.Tensor = Tensor
        root.constant = 1
        with patch.dict(sys.modules, {'torch':root}), patch('mirage.runtime.api_metadata.import_module') as importer:
            self.assertIs(resolve('torch.operation'), root.operation)
            self.assertIs(resolve('Tensor.method'), Tensor.method)
            self.assertIs(resolve('torch.Tensor.method'), Tensor.method)
            with self.assertRaises(AttributeError):
                resolve('Tensor.missing')
            with self.assertRaisesRegex(ValueError, 'not callable'):
                resolve('torch.constant')
            importer.assert_not_called()

    def test_missing_requested_submodule_restores_original_attribute_error(self):
        import sys
        from types import ModuleType
        from unittest.mock import patch
        from mirage.runtime.api_metadata import resolve
        missing = AttributeError('public lazy attribute absent')
        class Root(ModuleType):
            def __getattr__(self, name): raise missing
        root = Root('torch')
        error = ModuleNotFoundError('requested module missing', name='torch.absent')
        with patch.dict(sys.modules, {'torch':root}), patch(
                'mirage.runtime.api_metadata.import_module', side_effect=error) as importer:
            with self.assertRaises(AttributeError) as caught:
                resolve('torch.absent.operation')
            self.assertIs(caught.exception, missing)
            importer.assert_called_once_with('torch.absent')

    def test_missing_internal_dependency_and_other_import_errors_are_preserved(self):
        import sys
        from types import ModuleType
        from unittest.mock import patch
        from mirage.runtime.api_metadata import resolve
        root = ModuleType('torch')
        for error in (ModuleNotFoundError('internal dependency missing', name='dependency'),
                      ModuleNotFoundError('internal Torch dependency missing', name='torch.other'),
                      ImportError('initialization failed')):
            with self.subTest(error=error), patch.dict(sys.modules, {'torch':root}), patch(
                    'mirage.runtime.api_metadata.import_module', side_effect=error):
                with self.assertRaises(type(error)) as caught:
                    resolve('torch.lazy.operation')
                self.assertIs(caught.exception, error)

    def test_invalid_public_paths_and_external_or_private_module_aliases_do_not_import(self):
        import sys
        from types import ModuleType
        from unittest.mock import patch
        from mirage.runtime.api_metadata import resolve
        root = ModuleType('torch')
        root.external = ModuleType('outside')
        root.private = ModuleType('torch._private')
        root.notmodule = object()
        with patch.dict(sys.modules, {'torch':root}), patch('mirage.runtime.api_metadata.import_module') as importer:
            for api in ('outside.operation', 'torch._private.operation', 'torch.lazy..operation'):
                with self.subTest(api=api), self.assertRaises(ValueError):
                    resolve(api)
            for api in ('torch.external.lazy.operation', 'torch.private.lazy.operation', 'torch.notmodule.operation'):
                with self.subTest(api=api), self.assertRaises(AttributeError):
                    resolve(api)
            importer.assert_not_called()

    def test_installed_public_lazy_target_identity_and_metadata_are_preserved(self):
        from importlib import import_module
        from mirage.runtime.api_metadata import resolve
        target = resolve('torch.utils.checkpoint.checkpoint')
        self.assertIs(target, import_module('torch.utils.checkpoint').checkpoint)
        metadata = describe('torch.utils.checkpoint.checkpoint')
        self.assertEqual(metadata['api'], 'torch.utils.checkpoint.checkpoint')
        self.assertEqual(metadata['kind'], 'callable')
        self.assertTrue(metadata['documentation_sha256'])
        self.assertIn('function', metadata['signature'])


class GeneralProgramTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.initial = self.root / "initial.pt"
        initialize(SOURCE, configuration(), self.initial)
        self.counter = 0

    def tearDown(self):
        self.temp.cleanup()

    def execute(self, **kwargs):
        self.counter += 1
        return run(SOURCE, configuration(**kwargs), self.initial,
                   Recorder(self.root / f"trace-{self.counter}.jsonl"))

    def test_signature_and_open_api_resolution(self):
        info = describe("torch.sin")
        self.assertIsNotNone(info["signature"])
        self.assertIn("torch_version", info)

    def test_api_outside_acceptance_list_executes(self):
        source = SOURCE.replace('torch.matmul(ctx["inputs"]["x"], ctx["parameters"]["w"])',
                                'torch.sin(ctx["parameters"]["w"])')
        result = run(source, configuration(api="torch.sin"), self.initial,
                     Recorder(self.root / "sin.jsonl"))
        self.assertTrue(result["api_executed"])
        self.assertTrue(result["backward_executed"])

    def test_explicit_forward_only_target(self):
        source = SOURCE.replace('return torch.matmul(ctx["inputs"]["x"], ctx["parameters"]["w"])',
                                'return {"shape": ctx["inputs"]["x"].size(), "value": torch.matmul(ctx["inputs"]["x"], ctx["parameters"]["w"])}')
        source = source.replace('return output.square().mean()', 'return output["value"].square().mean()')
        config = configuration(api="Tensor.size", requires_backward=False)
        left = run(source, config, self.initial, Recorder(self.root / "size-source.jsonl"))
        right = run(source, dict(config, representation="strided"), self.initial,
                    Recorder(self.root / "size-followup.jsonl"))
        self.assertTrue(left["api_executed"])
        self.assertFalse(left["backward_executed"])
        self.assertEqual(inspect("layout", left, right)["verdict"], "pass")

    def test_lifecycle_contract(self):
        self.assertEqual(analyze(SOURCE)["version"], 3)
        with self.assertRaises(ContractError):
            analyze("def forward(x): return x")

    def test_layout(self):
        left, right = self.execute(), self.execute(representation="strided")
        self.assertTrue(right["api_executed"])
        self.assertTrue(right["backward_executed"])
        self.assertEqual(inspect("layout", left, right)["verdict"], "pass")

    def test_storage(self):
        self.assertEqual(inspect("storage", self.execute(), self.execute(representation="clone"))["verdict"], "pass")

    def test_checkpoint(self):
        self.assertEqual(inspect("checkpoint", self.execute(), self.execute(checkpoint=True))["verdict"], "pass")

    def test_saved_tensor_roundtrip(self):
        left, right = self.execute(), self.execute(saved_tensors=True)
        self.assertGreater(right["saved_tensors"]["unpacked"], 0)
        self.assertEqual(inspect("saved_tensor", left, right)["verdict"], "pass")

    def test_uneven_microbatches(self):
        left = self.execute(accumulation_source=True)
        right = self.execute(microbatches=2)
        self.assertTrue(left["batch_separable"])
        self.assertTrue(right["batch_separable"])
        self.assertNotEqual(left["backwards"], right["backwards"])
        self.assertEqual(inspect("accumulation", left, right)["verdict"], "pass")

    def test_history_and_retained_graph(self):
        result = self.execute(optimizer_steps=2, gradient_state="accumulated", graph_mode="retained")
        self.assertEqual(result["updates"], 3)
        self.assertEqual(result["backwards"], 5)

    def test_corrupted_output_is_suspected(self):
        left, right = self.execute(), self.execute(saved_tensors=True)
        right["observations"]["parameters"]["w"] += 10
        self.assertEqual(inspect("saved_tensor", left, right)["verdict"], "suspected_failure")

    def test_missing_evidence_is_not_pass(self):
        left, right = self.execute(), self.execute(saved_tensors=True)
        right["api_executed"] = False
        self.assertEqual(inspect("saved_tensor", left, right)["verdict"], "invalid")

    def test_unequal_initial_state_is_inapplicable(self):
        left, right = self.execute(), self.execute(saved_tensors=True)
        right["initial"]["inputs"]["x"] += 1
        self.assertEqual(inspect("saved_tensor", left, right)["verdict"], "inapplicable")

    def test_missing_saved_tensors_is_inapplicable(self):
        left, right = self.execute(), self.execute(saved_tensors=True)
        right["saved_tensors"]["unpacked"] = 0
        self.assertEqual(inspect("saved_tensor", left, right)["verdict"], "inapplicable")

    def test_nonfinite_is_unassessed(self):
        left, right = self.execute(), self.execute(saved_tensors=True)
        right["observations"]["parameters"]["w"][0, 0] = float("inf")
        self.assertIsNone(inspect("saved_tensor", left, right)["verdict"])

    def test_layout_must_reach_target_operands(self):
        source = SOURCE.replace('ctx["inputs"]["x"], ctx["parameters"]',
                                'ctx["inputs"]["x"].contiguous(), ctx["parameters"]')
        result = run(source, configuration(representation="strided", input_name="x"), self.initial,
                     Recorder(self.root / "materialized.jsonl"))
        self.assertFalse(result["target_state_reached"])

    def test_compiled_target_and_backward(self):
        result = self.execute(backend="compile_eager")
        self.assertTrue(result["api_executed"])
        self.assertTrue(result["backward_executed"])

    def test_compiled_detach_does_not_fake_target_backward(self):
        source = SOURCE.replace('return torch.matmul(ctx["inputs"]["x"], ctx["parameters"]["w"])',
                                'return torch.matmul(ctx["inputs"]["x"], ctx["parameters"]["w"]).detach() + ctx["parameters"]["w"].sum()')
        result = run(source, configuration(backend="compile_eager"), self.initial,
                     Recorder(self.root / "compiled-detach.jsonl"))
        self.assertTrue(result["api_executed"])
        self.assertFalse(result["backward_executed"])

    def test_compiled_unrelated_output_does_not_prove_target_backward(self):
        source = SOURCE.replace('return torch.matmul(ctx["inputs"]["x"], ctx["parameters"]["w"])',
                                'return (torch.matmul(ctx["inputs"]["x"], ctx["parameters"]["w"]), ctx["parameters"]["w"] * 2)')
        source = source.replace('return output.square().mean()', 'return output[1].square().mean()')
        result = run(source, configuration(backend="compile_eager"), self.initial,
                     Recorder(self.root / "compiled-unrelated.jsonl"))
        self.assertTrue(result["api_executed"])
        self.assertFalse(result["backward_executed"])

    def test_initial_input_gradients_survive_restore(self):
        import torch
        from mirage.generator.program import load
        from mirage.runtime.program import restore

        source = SOURCE.replace('    w = torch.nn.Parameter',
                                '    x.requires_grad_()\n    x.grad = torch.ones_like(x)\n    w = torch.nn.Parameter')
        initialize(source, configuration(), self.initial)
        restored = restore(load(source), configuration(), self.initial)
        self.assertTrue(torch.equal(restored["inputs"]["x"].grad, torch.ones(5, 3)))

    def test_inplace_keyword_is_observed_as_mutation(self):
        source = SOURCE.replace('torch.matmul(ctx["inputs"]["x"], ctx["parameters"]["w"])',
                                'torch.nn.functional.relu(ctx["parameters"]["w"] * 1.0, inplace=True)')
        result = run(source, configuration(api="torch.nn.functional.relu"), self.initial,
                     Recorder(self.root / "inplace-keyword.jsonl"))
        self.assertFalse(result["pure"])
        self.assertIn("relu", result["mutated_operations"])

    def test_mutation_rule(self):
        source = SOURCE.replace('torch.randn(5, 3', 'torch.randn(3, 2')
        source = source.replace('return torch.matmul(ctx["inputs"]["x"], ctx["parameters"]["w"])',
                                'value = ctx["parameters"]["w"] * 1.0\n    return value.add_(ctx["inputs"]["x"])')
        config = configuration(api="Tensor.add_")
        initial = self.root / "mutation.pt"
        initialize(source, config, initial)
        left = run(source, config, initial, Recorder(self.root / "mutation-eager.jsonl"))
        right = run(source, dict(config, backend="compile_eager"), initial,
                    Recorder(self.root / "mutation-compiled.jsonl"))
        self.assertEqual(inspect("mutation", left, right)["verdict"], "pass")

    def test_plan_order_and_actual_history(self):
        from mirage.contracts.exploration import validate_history

        config = configuration(optimizer_steps=1, gradient_state="zero")
        with self.assertRaises(ContractError):
            validate_history(["step", "zero:zero"], config)
        actions = ["zero:none", "forward", "backward:retain", "backward", "step", "zero:zero"]
        validate_history(actions, config)
        result = self.execute(optimizer_steps=1, gradient_state="zero", history=actions)
        self.assertTrue(result["target_state_reached"])
        self.assertEqual(result["updates"], 2)

    def test_cross_sample_matmul_is_not_certified(self):
        from mirage.runtime.mr_actions import accumulation_supported
        from mirage.generator.program import load

        source = SOURCE.replace('ctx["parameters"]["w"]', 'ctx["inputs"]["x"]')
        ctx = load(SOURCE)["build"]("cpu", 42)
        self.assertFalse(accumulation_supported(source, ctx, []))

    def test_signature_import_alias_is_legal(self):
        self.assertEqual(analyze("import torch.nn.functional as F\n" + SOURCE)["version"], 3)

    def test_module_parameters_are_rebound_after_restore(self):
        import torch
        source = '''
import torch
def build(device, seed):
    torch.manual_seed(seed)
    model = torch.nn.Linear(3, 2).to(device)
    return {"inputs": {"x": torch.randn(5, 3, device=device)},
            "parameters": dict(model.named_parameters()), "modules": {"model": model},
            "optimizer": torch.optim.Adam(model.parameters()), "batch_axes": {"x": 0}}
def forward(ctx):
    return ctx["modules"]["model"](ctx["inputs"]["x"])
def loss(output):
    return output.square().mean()
'''
        config = configuration(api="torch.nn.functional.linear")
        initial = self.root / "module.pt"
        initialize(source, config, initial)
        result = run(source, config, initial, Recorder(self.root / "module.jsonl"))
        self.assertTrue(result["backward_executed"])
        self.assertFalse(torch.equal(result["initial"]["state"]["parameters"]["weight"],
                                     result["observations"]["parameters"]["weight"]))

    def test_cuda_precision_and_inductor(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        config = configuration(device="cuda")
        initial = self.root / "cuda.pt"
        # Keep this equivalence fixture away from GradScaler overflow; nonfinite
        # observations have a separate unassessed-policy test.
        source = SOURCE.replace("torch.randn(", "0.05 * torch.randn(")
        initialize(source, config, initial)
        for precision in ("fp32", "fp16", "bf16"):
            with self.subTest(precision=precision):
                left = run(source, dict(config, precision=precision), initial,
                           Recorder(self.root / f"{precision}-source.jsonl"))
                right = run(source, dict(config, precision=precision, saved_tensors=True), initial,
                            Recorder(self.root / f"{precision}-followup.jsonl"))
                self.assertEqual(inspect("saved_tensor", left, right)["verdict"], "pass")
        for backend in ("aot_eager", "inductor"):
            with self.subTest(backend=backend):
                result = run(source, dict(config, backend=backend), initial,
                             Recorder(self.root / f"{backend}.jsonl"))
                self.assertTrue(result["api_executed"])
                self.assertTrue(result["backward_executed"])

    def test_accumulation_rejects_different_loss_normalization(self):
        source = SOURCE.replace('return output.square().mean()', 'return output.square().sum()')
        left = run(source, configuration(accumulation_source=True), self.initial,
                   Recorder(self.root / "wrong-loss-source.jsonl"))
        right = run(source, configuration(microbatches=2), self.initial,
                    Recorder(self.root / "wrong-loss-followup.jsonl"))
        self.assertEqual(inspect("accumulation", left, right)["verdict"], "inapplicable")


class StatefulFactoryGenerationContractTests(unittest.TestCase):
    def test_callable_protocol_is_selected_without_supplying_an_operator_or_inputs(self):
        from mirage.generator.program import stateful_request
        for target, kind in [('torch.compile', 'compiled_callable_factory'),
                             ('torch.cuda.make_graphed_callables', 'captured_callable_factory')]:
            with self.subTest(target=target):
                request = stateful_request({'documentation': 'Installed public specification'},
                    {'api': target, 'device': 'cuda', 'requires_backward': True})
                protocol = request.context['callable_protocol']
                self.assertEqual(protocol['kind'], kind)
                self.assertEqual(protocol['factory_target'], target)
                self.assertEqual(protocol['backward_scope'], 'returned_callable_boundary')
                self.assertIn('invoke_returned_callable', protocol['lifecycle'])
                self.assertNotIn('source', protocol)
                self.assertNotIn('inputs', protocol)
                self.assertIn('actual named_parameters', request.prompt)
        ordinary = stateful_request({'documentation': ''}, {'api': 'torch.cos', 'device': 'cpu'})
        self.assertIsNone(ordinary.context['callable_protocol'])


class StatefulTopLevelModuleDeclarationTests(unittest.TestCase):
    def source(self, declaration):
        return ("import torch\nfrom torch import nn\n" + declaration +
                "\ndef build(device, seed): return {'inputs': {}, 'parameters': {}}\n"
                "def run(ctx, probe): return {}\n")

    def test_inert_module_declarations_and_literal_defaults_are_accepted(self):
        from mirage.generator.program import analyze_stateful
        source = self.source('class Net(nn.Module):\n    "A module declaration."\n'
                             '    def forward(self, value, scale=1): return value * scale\n')
        self.assertEqual(analyze_stateful(source)['version'], 'stateful-1')
        self.assertEqual(analyze_stateful(self.source('class Net(torch.nn.Module):\n    pass\n'))['version'], 'stateful-1')

    def test_class_body_effects_decorators_metaclasses_and_other_bases_are_rejected(self):
        from mirage.generator.program import analyze_stateful
        declarations = [
            'class Net(nn.Module):\n    value = torch.ones(2)\n',
            'class Net(nn.Module):\n    torch.set_grad_enabled(False)\n',
            '@torch.no_grad()\nclass Net(nn.Module):\n    pass\n',
            'class Net(nn.Module, metaclass=type):\n    pass\n',
            'class Net(object):\n    pass\n',
            'class Net(nn.Module):\n    @torch.no_grad()\n    def forward(self, x): return x\n',
            'class Net(nn.Module):\n    def forward(self, x=torch.ones(2)): return x\n',
            'class Net(nn.Module):\n    def forward(self, x): return x\n    def forward(self, x): return x\n',
        ]
        for declaration in declarations:
            with self.subTest(declaration=declaration), self.assertRaises(ContractError):
                analyze_stateful(self.source(declaration))

    def test_top_level_module_admission_preserves_rng_globals_and_duplicate_checks(self):
        from mirage.generator.program import analyze_stateful
        declarations = [
            'class Net(nn.Module):\n    def forward(self, x): return torch.randn(2)\n',
            'class Net(nn.Module):\n    def forward(self, x):\n        global state\n        state = x\n        return x\n',
            'class Net(nn.Module):\n    pass\nclass Net(nn.Module):\n    pass\n',
            'class Net(nn.Module):\n    pass\ndef Net(): return None\n',
        ]
        for declaration in declarations:
            with self.subTest(declaration=declaration), self.assertRaises(ContractError):
                analyze_stateful(self.source(declaration))


class SourceOperandGuidanceTests(unittest.TestCase):
    def bind(self, body, api='torch.add', selectors=()):
        from mirage.generator.program import source_operand_bindings
        source = 'import torch\ndef run(ctx, probe):\n' + '\n'.join('    ' + line for line in body.splitlines()) + '\n'
        return source_operand_bindings(source, api, selectors)

    def test_bound_receiver_unbound_and_keyword_paths_match_runtime_convention(self):
        bound = self.bind('x = ctx["x"]\ny = probe.call(x.add, other=ctx["other"])', 'Tensor.add')
        paths = {tuple(item['operand_path']):item for item in bound['bindings']}
        self.assertEqual(paths['args',0]['expression'], 'x')
        self.assertTrue(paths['args',0]['receiver'])
        self.assertEqual(paths['kwargs','other']['expression'], "ctx['other']")
        direct = self.bind('y = probe.call(torch.Tensor.add, ctx["x"], ctx["other"])', 'torch.Tensor.add')
        self.assertEqual([item['operand_path'] for item in direct['bindings']], [['args',0], ['args',1]])
        self.assertTrue(all(not item['receiver'] for item in direct['bindings']))
        self.assertIn('Static source guidance only', bound['scope'])
        mismatched = self.bind('y = probe.call(torch.add, x, x)', 'Tensor.add')
        self.assertNotEqual(mismatched['status'], 'mapped_static')
        self.assertTrue(all(item['status']=='unresolved' for item in mismatched['bindings']))

    def test_import_alias_and_nested_literal_operand_paths(self):
        from mirage.generator.program import source_operand_bindings
        source = 'from torch import cat as combine\ndef run(ctx, probe):\n    y = probe.call(combine, [ctx["x"], ctx["y"]])\n'
        result = source_operand_bindings(source, 'torch.cat', [{'call_index':0, 'operand_path':['args',0,1]}])
        self.assertEqual(result['bindings'][0]['expression'], "ctx['y']")
        self.assertEqual(len(result['source_sha256']), 64)
        nested = self.bind('y = probe.call(torch.add, values, other=extras)', selectors=[
            {'call_index':0, 'operand_path':['args',0,1], 'peer_operand_path':['kwargs','other','right']}])
        self.assertEqual([item['expression'] for item in nested['bindings']], ['values[1]', "extras['right']"])

    def test_later_reassignment_is_exposed_without_changing_mutation_anchors(self):
        result = self.bind('x = ctx["x"]\na = probe.call(torch.add, x, x)\nx = ctx["next"]\nb = probe.call(torch.add, x, x)',
            selectors=[{'call_index':1, 'operand_path':['args',0]}])
        item = result['bindings'][0]
        self.assertEqual(item['last_definitions'][0]['statement_index'], 2)
        self.assertTrue(item['redefined_after_first_setup'])
        self.assertEqual(result['calls'][1]['first_operand_setup_statement_index'], 1)
        self.assertEqual(item['expression'], 'x')

    def test_ambiguous_dynamic_factory_and_missing_paths_stay_unresolved(self):
        for body in ('for x in ctx["xs"]:\n    y = probe.call(torch.add, x, x)',
                     'y = condition and probe.call(torch.add, x, x)',
                     'y = probe.call(torch.add, *values)', 'fn = torch.add\ny = probe.call(fn, x, x)',
                     'fn = probe.call(torch.compile, model)\ny = probe.call(fn, x)'):
            result = self.bind(body, selectors=[{'call_index':0, 'operand_path':['args',0]}])
            self.assertNotEqual(result['bindings'][0]['status'], 'mapped_static')
        from mirage.generator.program import source_operand_bindings
        helper = 'import torch\ndef helper(p, x): return p.call(torch.add, x, x)\ndef run(ctx, probe):\n    a = helper(probe, ctx["x"])\n    b = probe.call(torch.add, a, a)\n'
        self.assertEqual(source_operand_bindings(helper, 'torch.add')['status'], 'unresolved')
        missing = self.bind('y = probe.call(torch.add, x, x)', selectors=[
            {'call_index':0, 'operand_path':['args',8]}, {'call_index':4, 'operand_path':['args',0]}])
        self.assertTrue(all(item['status']=='unresolved' for item in missing['bindings']))

    def test_factory_symbol_aliases_leave_call_ordinals_unresolved(self):
        selectors = [{'call_index':0, 'operand_path':['args',0]}]
        for factory in ('torch.compile', 'torch.cuda.make_graphed_callables'):
            with self.subTest(factory=factory):
                result = self.bind('factory = ' + factory + '\nwrapped = factory(ctx["model"])\n'
                    'prior = wrapped(ctx["u"])\ny = probe.call(torch.add, ctx["v"], ctx["v"])', selectors=selectors)
                self.assertEqual(result['status'], 'unresolved')
                self.assertEqual(result['bindings'][0]['status'], 'unresolved')
                self.assertIn('runtime call-site linkage', result['reason'])
        from mirage.generator.program import source_operand_bindings
        for declaration, expression in (
                ('from torch import compile as factory', 'factory'),
                ('from torch.cuda import make_graphed_callables as factory', 'factory'),
                ('import torch as framework', 'framework.compile')):
            with self.subTest(declaration=declaration):
                result = self.bind(declaration + '\nfactory_alias = ' + expression +
                    '\ny = probe.call(torch.add, x, x)', selectors=selectors)
                self.assertEqual(result['status'], 'unresolved')
                source = declaration + '\ndef run(ctx, probe):\n    y = probe.call(torch.add, x, x)\n'
                if expression == 'factory':
                    self.assertEqual(source_operand_bindings(source, 'torch.add', selectors)['status'], 'unresolved')

    def test_probe_rebinding_leaves_source_bindings_unresolved(self):
        for binding in ('probe = ctx["other_probe"]', 'del probe',
                        'import contextlib as probe', 'def probe():\n    return None',
                        'class probe:\n    pass'):
            with self.subTest(binding=binding):
                result = self.bind(binding + '\ny = probe.call(torch.add, x, x)', selectors=[
                    {'call_index':0, 'operand_path':['args',0]}])
                self.assertEqual(result['status'], 'unresolved')
                self.assertEqual(result['bindings'][0]['status'], 'unresolved')
                self.assertIn('probe binding', result['reason'])
                self.assertIn('Static source guidance only', result['scope'])

    def test_training_predicate_guidance_does_not_require_operand_binding(self):
        from mirage.generator.program import stateful_request
        source = 'import torch\ndef build(device, seed): return {"inputs":{},"parameters":{}}\ndef run(ctx, probe):\n    with probe.call(torch.no_grad):\n        value = torch.ones(1)\n    return {"value":value}\n'
        request = stateful_request({'documentation':'installed spec'},
            {'api':'torch.no_grad','device':'cpu','requires_backward':False}, canonical=source,
            predicates=[{'dimension':'context','field':'restoration_count','expected':2,'checkpoint':'final'}])
        self.assertEqual(request.context['source_operand_bindings']['status'], 'unresolved')
        self.assertIn('only predicates selecting operand paths', request.prompt)
        self.assertIn('without operand paths do not require tensor bindings', request.prompt)
        self.assertNotIn('TENSOR RELATION GOALS', request.prompt)

    def test_bindings_and_diagnostic_feedback_are_bounded_and_json_serializable(self):
        import json
        from mirage.generator.program import unmet_predicate_feedback
        result = self.bind('\n'.join('v%d = probe.call(torch.add, x, x)' % i for i in range(20)))
        self.assertEqual(len(result['calls']), 8)
        self.assertEqual(result['omitted_calls'], 12)
        self.assertEqual(len(result['bindings']), 24)
        self.assertEqual(result['omitted_bindings'], 16)
        predicates = [{'call_index':0, 'operand_path':['args',0], 'dimension':'representation',
            'field':'contiguous', 'expected':False, 'observed':True, 'attained':False,
            'evidence_status':'verified', 'reason':'x' * 2000} for _ in range(5)]
        inspection = {'api_executed':True, 'predicates':predicates,
                      'semantic_coverage':{'large':['never include full coverage'] * 100}}
        feedback = unmet_predicate_feedback(inspection, bindings=result, limit=2)
        self.assertEqual(len(feedback['unmet_predicates']), 2)
        self.assertEqual(feedback['omitted_predicates'], 3)
        self.assertEqual(len(feedback['unmet_predicates'][0]['reason']), 600)
        self.assertEqual(feedback['source_bindings'][0]['expression'], 'x')
        self.assertNotIn('semantic_coverage', feedback)
        json.dumps(feedback, allow_nan=False)
        self.assertEqual(len(inspection['predicates']), 5)


class OptimizerGenerationGuidanceTests(unittest.TestCase):
    def test_fresh_backward_guidance_preserves_legal_nonoptimizer_coverage(self):
        from mirage.generator.program import stateful_request
        request = stateful_request({'documentation': 'installed spec'}, {'api': 'torch.sin', 'requires_backward': True})
        for text in ('ATTRIBUTABLE UPDATE', 'SAME object', 'foreach=False', 'fused=False',
                     'source', 'context-only or callable-factory', 'retain legal API coverage'):
            self.assertIn(text, request.prompt)
        forward = stateful_request({}, {'api': 'torch.sin', 'requires_backward': False})
        self.assertNotIn('ATTRIBUTABLE UPDATE', forward.prompt)
        self.assertEqual(request.context['prompt_version'], 'stateful-10-edges')

    def test_gradient_goal_guidance_and_repair_keep_parameter_identity(self):
        from mirage.generator.program import stateful_request, unmet_predicate_feedback
        predicate = {'dimension': 'gradient_representation', 'field': 'contiguous', 'expected': False,
            'checkpoint': 'optimizer:0/step:0/pre_update', 'parameter_key': 'weight',
            'call_index': 0, 'operand_path': ['args', 0], 'relation': 'target_gradient_update'}
        request = stateful_request({}, {'api': 'torch.sin', 'requires_backward': True}, predicates=[predicate])
        for text in ('optimizer_gradient_action', 'do not assign p.grad', 'companion predicates',
                     'not backward accumulation', 'never emit probe.mark'):
            self.assertIn(text, request.prompt)
        feedback = unmet_predicate_feedback({'predicates': [dict(predicate, attained=False)]})
        self.assertEqual(feedback['unmet_predicates'][0]['parameter_key'], 'weight')


if __name__ == "__main__":
    unittest.main()


class RepresentationTraversalPromptTests(unittest.TestCase):
    def test_framework_owns_input_representation_only_in_traversal_treatment(self):
        from mirage.generator.program import stateful_request
        config = {'api': 'torch.sin', 'device': 'cpu', 'requires_backward': True}
        legacy = stateful_request({}, config)
        traversal = stateful_request({}, dict(config, representation_traversal=True))
        self.assertNotIn('FRAMEWORK REPRESENTATION TRAVERSAL', legacy.prompt)
        self.assertIn('FRAMEWORK REPRESENTATION TRAVERSAL', traversal.prompt)
        self.assertIn('semantic input values/shapes', traversal.prompt)
        self.assertIn('do not add storage-layout tricks', traversal.prompt)

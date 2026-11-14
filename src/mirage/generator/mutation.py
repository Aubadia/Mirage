"""Bounded statement replacements in immutable generated parent programs."""
import ast
from copy import deepcopy
import hashlib

from mirage.generator.program import ContractError, analyze_stateful

VERSION = 'local-statements-1'


def source_hash(source):
    return hashlib.sha256(source.encode()).hexdigest()


def _target_calls(node):
    return [ast.dump(n, include_attributes=False) for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and isinstance(n.func.value, ast.Name) and n.func.value.id == 'probe'
            and n.func.attr == 'call']


def plan_mutation(source, parent_id, attempt, *, kind=None):
    """Choose a small editable region; unsupported structures use fresh generation."""
    analyze_stateful(source)
    tree = ast.parse(source)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    run, build = functions['run'], functions['build']
    anchors = [i for i, statement in enumerate(run.body)
               if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.Expr)) and _target_calls(statement)]
    if not anchors or len(_target_calls(tree)) != len(anchors):
        return None
    if any(isinstance(n, (ast.Return, ast.Raise)) for statement in run.body[:anchors[-1]]
           for n in ast.walk(statement)):
        return None
    kinds = ('run_tail', 'operand_setup', 'initialization')
    selected = kind or kinds[attempt % len(kinds)]
    if selected not in kinds:
        raise ValueError('Unknown local mutation kind')
    if selected == 'run_tail':
        function, start, stop = 'run', anchors[-1] + 1, len(run.body)
    elif selected == 'operand_setup':
        function, start, stop = 'run', anchors[0], anchors[0]
    else:
        candidates = [i for i, statement in enumerate(build.body)
                      if isinstance(statement, ast.Assign) and all(isinstance(t, ast.Name) for t in statement.targets)
                      and not _target_calls(statement)]
        if not candidates:
            return None
        function, start = 'build', candidates[attempt % len(candidates)]
        stop = start + 1
    original = functions[function].body[start:stop]
    if len(original) > 8 or sum(sum(1 for _ in ast.walk(n)) for n in original) > 128:
        return None
    return {'version': VERSION, 'parent_id': parent_id, 'parent_sha256': source_hash(source),
            'kind': selected, 'protected_call_indices': list(range(len(anchors))), 'editable_region': {'function': function, 'start': start, 'stop': stop},
            'original_statements': [ast.unparse(n) for n in original],
            'region_statements': [{'index': i, 'source': ast.unparse(n)}
                                  for i, n in enumerate(functions[function].body)],
            'budget': {'characters': 4096, 'statements': 8, 'ast_nodes': 128},
            'preserved_obligations': ['unchanged AST outside editable region', 'all existing probe.call invocations',
                                      'configured target execution', 'required target backward path'],
            'allowed_semantic_changes': ['local training history and observables' if function == 'run'
                                         else 'explicit initialization shared by the child MR members'],
            'parent_child_relation': 'exploration lineage only; revalidate the child MR pair'}


def apply_mutation(parent, replacement, contract):
    """Apply only the declared region, then verify the assembled static contract."""
    if contract.get('version') != VERSION or source_hash(parent) != contract.get('parent_sha256'):
        raise ContractError('Mutation parent/version mismatch')
    if not isinstance(replacement, str) or not replacement.strip():
        raise ContractError('Mutation source must contain replacement statements')
    budget = contract['budget']
    if len(replacement) > budget['characters']:
        raise ContractError('Mutation character budget exceeded')
    patch = ast.parse(replacement)
    if len(patch.body) > budget['statements'] or sum(1 for _ in ast.walk(patch)) > budget['ast_nodes']:
        raise ContractError('Mutation statement/node budget exceeded')
    if any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in ('build', 'run')
           for n in ast.walk(patch)):
        raise ContractError('Return replacement statements, not complete build/run functions')
    if contract['kind'] != 'run_tail' and any(
            isinstance(n, (ast.Return, ast.Raise, ast.Break, ast.Continue)) or
            isinstance(n, ast.Call) and _target_calls(n) for n in ast.walk(patch)):
        raise ContractError('Setup/initialization edits cannot bypass or precede the protected target with another call')
    if any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(patch)):
        raise ContractError('Imports are outside the editable statement contract')
    tree = ast.parse(parent)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    region = contract['editable_region']
    function = functions.get(region['function'])
    start, stop = region['start'], region['stop']
    if function is None or not (type(start) is int and type(stop) is int and 0 <= start <= stop <= len(function.body)):
        raise ContractError('Invalid editable region')
    original = function.body[start:stop]
    if any(_target_calls(n) for n in original):
        raise ContractError('Editable region contains a protected target invocation')
    before = deepcopy(tree)
    function.body[start:stop] = patch.body
    # Strip the edited region from both trees to establish the static frame.
    expected = next(n for n in before.body if isinstance(n, ast.FunctionDef) and n.name == function.name)
    expected.body[start:stop] = [ast.Pass()]
    framed = deepcopy(tree)
    next(n for n in framed.body if isinstance(n, ast.FunctionDef) and n.name == function.name).body[
        start:start + len(patch.body)] = [ast.Pass()]
    if ast.dump(before, include_attributes=False) != ast.dump(framed, include_attributes=False):
        raise ContractError('Mutation changed AST outside its editable region')
    source = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    graph = analyze_stateful(source)
    if graph['fingerprint'] == analyze_stateful(parent)['fingerprint']:
        raise ContractError('Mutation made no semantic AST edit')
    return source, {'version': VERSION, 'parent_id': contract['parent_id'],
                    'parent_sha256': contract['parent_sha256'], 'child_sha256': source_hash(source),
                    'kind': contract['kind'], 'editable_region': deepcopy(region),
                    'replacement': replacement, 'outside_region_unchanged': True,
                    'runtime_obligations': 'not established by static editing'}


def plan_regions(source, parent_id, attempt, regions):
    """Regions share one immutable preimage; edits are applied simultaneously."""
    if 'full' in regions:
        return None
    contracts = [plan_mutation(source, parent_id, attempt, kind=kind) for kind in dict.fromkeys(regions)]
    if not contracts or any(c is None for c in contracts):
        return None
    for contract in contracts:
        region = contract['editable_region']
        contract['region_id'] = f"{region['function']}:{region['start']}:{region['stop']}"
        contract['preimage_sha256'] = source_hash('\n'.join(contract['original_statements']))
    return {'version': 'local-regions-2', 'parent_id': parent_id,
            'parent_sha256': source_hash(source), 'kind': 'multi_region', 'regions': contracts,
            'budget': {'characters': 8192, 'statements': 16, 'ast_nodes': 256}}


def apply_regions(parent, patches, contract):
    if contract.get('version') != 'local-regions-2' or source_hash(parent) != contract.get('parent_sha256'):
        raise ContractError('Mutation parent/version mismatch')
    if not isinstance(patches, list) or len(patches) != len(contract['regions']):
        raise ContractError('Return exactly one patch for every declared region')
    indexed = {}
    for patch in patches:
        if not isinstance(patch, dict) or set(patch) != {'region_id', 'preimage_sha256', 'source'}:
            raise ContractError('Invalid patch fields')
        if not isinstance(patch['region_id'], str) or patch['region_id'] in indexed:
            raise ContractError('Duplicate region patch')
        indexed[patch['region_id']] = patch
    tree = ast.parse(parent)
    functions = {n.name:n for n in tree.body if isinstance(n, ast.FunctionDef)}
    total_chars = total_nodes = total_statements = 0
    prepared = []
    for region in contract['regions']:
        patch = indexed.get(region['region_id'])
        if patch is None or patch['preimage_sha256'] != region['preimage_sha256']:
            raise ContractError('Region preimage mismatch')
        snippet = patch['source']
        if not isinstance(snippet, str):
            raise ContractError('Patch source must be a string')
        # Validate each frame against the original parent, allowing a region no-op
        # when another region carries the semantic change.
        try:
            apply_mutation(parent, snippet, region)
        except ContractError as error:
            if str(error) != 'Mutation made no semantic AST edit':
                raise
        parsed = ast.parse(snippet)
        total_chars += len(snippet); total_nodes += sum(1 for _ in ast.walk(parsed)); total_statements += len(parsed.body)
        prepared.append((region['editable_region'], parsed.body))
    budget = contract['budget']
    if total_chars > budget['characters'] or total_nodes > budget['ast_nodes'] or total_statements > budget['statements']:
        raise ContractError('Combined mutation budget exceeded')
    # Descending original indices preserve each region's immutable coordinates.
    for region, body in sorted(prepared, key=lambda item:(item[0]['function'], item[0]['start']), reverse=True):
        functions[region['function']].body[region['start']:region['stop']] = body
    child = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    if analyze_stateful(child)['fingerprint'] == analyze_stateful(parent)['fingerprint']:
        raise ContractError('Mutation made no semantic AST edit')
    return child, {'version': contract['version'], 'parent_id': contract['parent_id'],
                   'parent_sha256': contract['parent_sha256'], 'child_sha256': source_hash(child),
                   'kind': 'multi_region', 'patches': patches, 'outside_region_unchanged': True,
                   'runtime_obligations': 'not established by static editing'}


PATCH_SCHEMA = {'type': 'object', 'properties': {'patches': {'type': 'array', 'minItems': 1,
    'maxItems': 3, 'items': {'type': 'object', 'properties': {
        'region_id': {'type': 'string'}, 'preimage_sha256': {'type': 'string'}, 'source': {'type': 'string'}},
        'required': ['region_id', 'preimage_sha256', 'source'], 'additionalProperties': False}}},
    'required': ['patches'], 'additionalProperties': False}

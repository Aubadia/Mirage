"""Frozen, equal-budget stateful-search ablations using a prepared model service."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from mirage.artifacts.bundle import inventory
from mirage.artifacts.report import summarize_search
from mirage.contracts.programs import RunOptions
from mirage.workflow import run_stateful, replay_stateful

ARMS = (('full-transition', 'full', 'transition', 'adaptive'),
        ('local-transition', 'local', 'transition', 'adaptive'),
        ('full-semantic', 'full', 'semantic', 'adaptive'),
        ('local-semantic', 'local', 'semantic', 'adaptive'),
        ('local-semantic-round-robin', 'local', 'semantic', 'round_robin'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget', type=float, default=120)
    parser.add_argument('--cases', type=int, default=12)
    parser.add_argument('--seeds', type=int, nargs='+', default=[11,29])
    parser.add_argument('--apis', nargs='+', default=['torch.sin','torch.add'])
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve()
    arms = (ARMS[3],) if args.pilot else ARMS
    serving_path = Path(os.environ.get('MIRAGE_SESSION_DIR', '.'))/'serving/environment.json'
    serving = json.loads(serving_path.read_text()) if serving_path.is_file() else None
    protocol = {'version':'stateful-search-ablation-1', 'pilot':args.pilot,
                'apis':args.apis, 'seeds':args.seeds, 'arms':arms,
                'per_run_seconds':args.budget, 'case_cap':args.cases,
                'order':'forward for first seed, reverse for second; alternate thereafter',
                'coverage_refinement':False, 'state_predicates':[] if args.pilot else 'runtime-derived',
                'model':os.environ.get('MIRAGE_VLLM_MODEL'), 'serving_environment':serving,
                'max_tokens':os.environ.get('MIRAGE_MAX_TOKENS'),
                'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                'engine_inventory':inventory(source.parents[1]/'src'),
                'budget_scope':'search time and case caps; metadata and reproduction are reported separately',
                'limitations':'Small pilot-sized comparison; no claim of general superiority, statistical significance or confirmed defects.'}
    write(args.output/'protocol.json', protocol)
    results = []
    for block, seed in enumerate(args.seeds):
        for name, generation, coverage, policy in (arms if block % 2 == 0 else tuple(reversed(arms))):
            directory = args.output/f'seed-{seed}-{name}'
            print(f'START seed={seed} arm={name}', flush=True)
            started = time.monotonic()
            run = run_stateful(RunOptions(str(directory),'vllm',cases=args.cases,time_limit=args.budget,
                               worker_timeout=90,seed=seed,device='cuda',strategy='stateful',contract_version=4),
                               tuple(args.apis),predicates=[] if args.pilot else None,coverage_refinement=False,
                               search_policy=policy,generation_mode=generation,coverage_basis=coverage)
            search_wall = time.monotonic()-started
            result = {'seed':seed, 'arm':name, 'generation_mode':generation, 'coverage_basis':coverage,
                      'search_policy':policy, 'metrics':summarize_search(run), 'search_wall_seconds':search_wall,
                      'replay':None, 'replay_seconds':0.}
            if run['stop_reason'] != 'provider_cancellation_unconfirmed':
                eligible = next((r for r in run['records'] if r.get('inspection',{}).get('valid')
                                 and r.get('mutation_obligations_valid') is not False
                                 and (Path(r['directory'])/'stateful-manifest.json').exists()), None)
                if eligible:
                    replay_start = time.monotonic()
                    result['replay'] = replay_stateful(Path(eligible['directory'])/'stateful-manifest.json',timeout=90)
                    result['replay_seconds'] = time.monotonic()-replay_start
            results.append(result)
            write(args.output/'comparison.json', {'protocol':protocol, 'runs':results,
                 'complete':len(results)==len(arms)*len(args.seeds),
                 'interpretation':'Report per-run observed metrics; no model-efficiency advantage is assumed.'})
            print(json.dumps({k:v for k,v in result.items() if k != 'replay'}),flush=True)
            if run['stop_reason'] == 'provider_cancellation_unconfirmed':
                return 2
    return 1 if any(r['replay'] and not r['replay']['replay_valid'] for r in results) else 0


if __name__ == '__main__':
    raise SystemExit(main())

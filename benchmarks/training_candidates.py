"""Frozen training campaign over a benchmark roster, independent of sampling logic."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.request


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def inventory(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}


def prepare(args):
    if min(args.seconds, args.active_apis, args.workers, args.generation_concurrency) < 1:
        raise ValueError('Positive API budgets and scheduling limits required')
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=False)
    code = args.output / "code"
    code.mkdir()
    for name in ("src", "tests", "benchmarks", "sbatch"):
        shutil.copytree(root / name, code / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(root / "pyproject.toml", code / "pyproject.toml")
    roster = json.loads((root / "benchmarks/pytorch_correctness/candidates.json").read_text())
    by_id = {row["id"]: row for row in roster["candidates"]}
    targets = [{key: by_id[identifier][key] for key in ("id", "api", "device")}
               for identifier in roster["selected"]]
    if not targets or len({t["api"] for t in targets}) != len(targets):
        raise ValueError("Benchmark targets must be nonempty and distinct")
    protocol = dict(version="training-candidates-campaign-2", strategy="training",
        shared_scheduling=True,
        source_contract_version=3, candidate_contract_version="training-candidates-1",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        targets=targets, code=str(code.resolve()), code_sha256=inventory(code),
        wall_seconds_per_api=args.seconds, worker_timeout=120, workers=args.workers,
        generation_concurrency=args.generation_concurrency, active_api_limit=args.active_apis, canonicals_per_api=3, targets_per_canonical=12,
        paths_per_target=2, cases=100, seed=0, replay_failures=1,
        model="Qwen/Qwen3.8-27B", model_revision="37561816e66b5b9153b1b64c10968c18c1e6f4a0",
        max_model_len=16384, max_tokens=4096, mtp_tokens=3,
        temperature=0, top_p=1, created_epoch=time.time(),
        budget_scope="per-API shared framework wall time; service startup and child launch/report overhead recorded separately",
        scope="roster identities/devices only; issue fixtures, conditions and oracle hints never enter generation",
        generality="roster results do not prove all-API or cross-framework support")
    write(args.output / "protocol.json", protocol)
    print(args.output / "protocol.json", flush=True)


def verify(path):
    protocol = json.loads(path.read_text())
    if inventory(Path(protocol["code"])) != protocol["code_sha256"]:
        raise RuntimeError("Frozen campaign code changed")
    if not os.environ.get("SLURM_STEP_ID"):
        raise RuntimeError("Execution requires a Slurm step")
    return protocol


def service_idle():
    url = os.environ["MIRAGE_VLLM_BASE_URL"].removesuffix("/v1") + "/metrics"
    with urllib.request.urlopen(url, timeout=5) as response:
        lines = response.read().decode().splitlines()
    values = {line.rsplit(" ", 1)[0]: float(line.rsplit(" ", 1)[1]) for line in lines
              if line.startswith(("vllm:num_requests_running", "vllm:num_requests_waiting"))}
    present = all(any(k.startswith("vllm:" + metric) for k in values)
                  for metric in ("num_requests_running", "num_requests_waiting"))
    return {"epoch": time.time(), "values": values,
            "idle": present and all(v == 0 for v in values.values())}


def confirm_idle(path):
    deadline, consecutive, samples = time.monotonic() + 120, 0, []
    while time.monotonic() < deadline:
        row = service_idle()
        samples.append(row)
        consecutive = consecutive + 1 if row["idle"] else 0
        write(path, {"samples": samples, "confirmed": consecutive >= 2})
        if consecutive >= 2:
            return True
        time.sleep(2)
    return False


def one(args):
    protocol = verify(args.protocol)
    sys.path.insert(0, str(Path(protocol["code"]) / "src"))
    from mirage.contracts.programs import RunOptions
    from mirage.workflow import run
    target = protocol["targets"][args.index]
    directory = args.protocol.parent / f"api-{args.index:02d}-{target['api'].replace('.', '-')}"
    directory.mkdir(exist_ok=False)
    options = RunOptions(str(directory / "cases"), "vllm", device=target["device"],
        time_limit=protocol["wall_seconds_per_api"], strategy="training", **{key: protocol[key] for key in
        ("worker_timeout", "workers", "generation_concurrency", "canonicals_per_api", "targets_per_canonical",
         "paths_per_target", "cases", "seed", "replay_failures")})
    started = time.monotonic()
    write(directory / "started.json", {"epoch": time.time(), "target": target,
          "job_id": os.environ["SLURM_JOB_ID"], "step_id": os.environ["SLURM_STEP_ID"]})
    status = 1
    try:
        report = run(options, (target["api"],))
        write(directory / "report.json", report)
        status = 0
    finally:
        write(directory / "completion.json", {"exit_status": status, "finished_epoch": time.time(),
                                               "framework_seconds": time.monotonic() - started})
    return status


def parallel_campaign(protocol, root):
    from mirage.contracts.programs import RunOptions
    from mirage.contracts.scheduling import APIJob, SchedulingOptions
    from mirage.runtime.multi_api import run_sessions
    jobs = []
    for index, target in enumerate(protocol['targets']):
        directory = root / f"api-{index:02d}-{target['api'].replace('.', '-')}"
        options = RunOptions(str(directory / 'cases'), 'vllm', strategy='training',
            device=target['device'], time_limit=protocol['wall_seconds_per_api'], **{
            key: protocol[key] for key in ('worker_timeout', 'workers', 'generation_concurrency',
                'canonicals_per_api', 'targets_per_canonical', 'paths_per_target', 'cases', 'seed', 'replay_failures')})
        jobs.append(APIJob(f'api-{index:03d}', target['api'], options))
    limits = SchedulingOptions(protocol['workers'], protocol['generation_concurrency'], protocol['active_api_limit'])
    result = run_sessions(jobs, limits, root / 'parallel')
    for row in result['sessions']:
        directory = Path(row['directory']).parent
        write(directory / 'report.json', row.get('report'))
        write(directory / 'completion.json', {k: v for k, v in row.items() if k != 'report'})
    return result


def campaign(args):
    protocol = verify(args.protocol)
    sys.path.insert(0, str(Path(protocol["code"]) / "src"))
    session = Path(os.environ["MIRAGE_SESSION_DIR"])
    serving = json.loads((session / "serving/environment.json").read_text())
    for key in ("model", "max_model_len", "mtp_tokens"):
        if serving[key] != protocol[key]:
            raise ValueError("Serving differs from frozen protocol: " + key)
    if serving["declared_revision"] != protocol["model_revision"]:
        raise ValueError("Model revision mismatch")
    write(args.protocol.parent / "campaign-start.json", {"epoch": time.time(), "serving": serving,
          "job_id": os.environ["SLURM_JOB_ID"], "step_id": os.environ["SLURM_STEP_ID"]})
    started, rows, status = time.monotonic(), [], 1
    child = None

    def stop(signum, _frame):
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        # Check the real serving backend's finite-array schema before spending
        # any benchmark budget. This is protocol validation, not API coverage.
        from mirage.contracts.exploration import history_request, validate_history
        from mirage.generator.http_provider import HTTPProvider
        from mirage.sampler.candidates import legal_paths
        config = dict(optimizer_steps=1, gradient_state="none", seed=0)
        paths = legal_paths(config)
        before = time.monotonic()
        preflight_directory = args.protocol.parent / "path-schema-preflight"
        preflight_directory.mkdir(parents=True, exist_ok=True)
        response = HTTPProvider.from_environment(directory=preflight_directory).generate(
            history_request(config, "", 1, paths), deadline=time.monotonic() + 60)
        actions = json.loads(response.content)["actions"]
        if validate_history(actions, config) not in paths:
            raise ValueError("Service returned a history outside the offered catalogue")
        write(args.protocol.parent / "path-schema-preflight.json", {"seconds": time.monotonic() - before,
            "actions": actions, "offered_paths": paths, "usage": vars(response.usage)})
        if protocol.get('shared_scheduling') or protocol.get('active_api_limit', 1) > 1:
            if not confirm_idle(args.protocol.parent / 'service-before-parallel.json'):
                raise RuntimeError('Service is not idle before parallel campaign')
            result = parallel_campaign(protocol, args.protocol.parent)
            rows.extend(result['sessions'])
            if not confirm_idle(args.protocol.parent / 'service-after-parallel.json'):
                raise RuntimeError('Inference cancellation unconfirmed after parallel campaign')
            verify(args.protocol)
            status = int(result['stop_reason'] != 'completed')
            return status
        for index, target in enumerate(protocol["targets"]):
            if not confirm_idle(args.protocol.parent / f"service-before-{index:02d}.json"):
                raise RuntimeError("Service requests not independently idle; refusing overlapping API batches")
            command = [sys.executable, str(Path(__file__).resolve()), "one", "--protocol", str(args.protocol), "--index", str(index)]
            print("Starting training framework:", target["api"], target["device"], flush=True)
            with (args.protocol.parent / f"api-{index:02d}.stdout.log").open("w") as out, (args.protocol.parent / f"api-{index:02d}.stderr.log").open("w") as err:
                before = time.monotonic()
                child = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True)
                try:
                    code = child.wait(timeout=protocol["wall_seconds_per_api"] + 90)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    code = 124
                rows.append(dict(target=target, exit_status=code, child_seconds=time.monotonic() - before))
                child = None
            write(args.protocol.parent / "campaign-progress.json", rows)
            if not confirm_idle(args.protocol.parent / f"service-after-{index:02d}.json"):
                raise RuntimeError("Inference cancellation unconfirmed; ending campaign")
        verify(args.protocol)
        status = int(any(row["exit_status"] != 0 for row in rows))
        return status
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        write(args.protocol.parent / "campaign-completion.json", {"exit_status": status,
            "finished_epoch": time.time(), "campaign_seconds": time.monotonic() - started, "runs": rows})


def main():
    parser = argparse.ArgumentParser(__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seconds", type=int, default=500)
    p.add_argument("--active-apis", type=int, default=1)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--generation-concurrency", type=int, default=2)
    for name in ("campaign", "one"):
        p = commands.add_parser(name)
        p.add_argument("--protocol", type=Path, required=True)
        if name == "one":
            p.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    return {"prepare": prepare, "campaign": campaign, "one": one}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())

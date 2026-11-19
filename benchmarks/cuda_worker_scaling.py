"""Measure fixed archived differential work on one Slurm-visible CUDA device."""
import concurrent.futures as cf
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import statistics
import sys
import time
from collections import defaultdict
from contextlib import ExitStack

clock = time.monotonic
ARCHIVE = Path("/work/asciibase64/Mirage/results/stateful-check-e2e/805400")


def write(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(path)


def signature(result):
    return [result.get("status"), result.get("reason"),
            sorted(result.get("raw", {}).get("differences", [])),
            sorted(result.get("stable_observed_fields", []))]


def side_signature(side):
    return [side.get(k) for k in ("status", "reason", "target_returned", "completed",
                                  "requested_state_attained", "target_exception")]


def select():
    strata = defaultdict(list)
    sessions = json.loads((ARCHIVE / "benchmark-results.json").read_text())["sessions"]
    for session in sessions:
        path = Path(session["report"]).parent / "differential-ledger.json"
        for row in json.loads(path.read_text()):
            if not row.get("legal_member"):
                continue
            api, member = session["api"], row["member"]
            case = f"{api}-{row['attempt']}-{member}"
            sides = row["devices"]
            replay = "1-cuda" in sides
            comp = row.get("comparison") or {}
            item = dict(id=case, manifest=row["manifest"], replay=replay,
                        expected=signature(comp), sides={k:side_signature(v) for k,v in sides.items()})
            strata[api, member, replay, bool(comp.get("raw", {}).get("differences"))].append(item)
    def rank(item):
        return hashlib.sha256(("805400-v1-" + item["id"]).encode()).digest()
    for group in strata.values():
        group.sort(key=rank)
    chosen = [group[0] for group in strata.values()]
    rest = sorted((x for group in strata.values() for x in group[1:]), key=rank)
    chosen.extend(rest[:max(0, 96-len(chosen))])
    return sorted(chosen, key=rank)


def initialize(cuda, gate):
    import torch
    from mirage.inspector import comparison
    torch.set_num_threads(1)
    if cuda:
        assert torch.cuda.device_count() == 1
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
    gate.wait(timeout=240)


def ping():
    return os.getpid()


def side_task(item, device, gpu, destination):
    import torch
    from mirage.inspector.comparison import differential_worker
    directory = Path(destination)
    directory.mkdir(parents=True)
    start = clock()
    # Cases may intentionally compile in mid-program; reset before building the case.
    torch.compiler.reset()
    spec = json.loads(Path(item["manifest"]).read_text())
    request = dict(spec, device=device, source=Path(spec["source"]).read_text())
    rng = random.getstate(), torch.get_rng_state(), torch.cuda.get_rng_state() if gpu else None
    grad, dtype = torch.is_grad_enabled(), torch.get_default_dtype()
    cast = {k:torch.is_autocast_enabled(k) for k in ("cpu","cuda")}
    try:
        result = differential_worker(request, directory)
        if gpu:
            torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info() if gpu else (0, 0)
        return dict(sig=side_signature(result), pid=os.getpid(), seconds=clock()-start,
                    device_used=total-free, gpu=torch.cuda.get_device_name() if gpu else None)
    finally:
        random.setstate(rng[0])
        torch.set_rng_state(rng[1])
        if gpu:
            torch.cuda.set_rng_state(rng[2])
        torch.set_grad_enabled(grad)
        torch.set_default_dtype(dtype)
        for k, value in cast.items():
            torch.set_autocast_enabled(k, value)


def compare(item, root, wave):
    from mirage.inspector.comparison import compare_saved_devices
    spec = json.loads(Path(item["manifest"]).read_text())
    paths = [str(root / item["id"] / f"{idx}-{device}" / "member.pt")
             for idx in range(wave+1) for device in ("cpu","cuda")]
    return signature(compare_saved_devices(dict(spec, members=paths)))


def run_wave(root, items, count, wave, results):
    ctx = mp.get_context("spawn")
    start = clock()
    with ExitStack() as stack:
        def pool(n, cuda):
            p = stack.enter_context(cf.ProcessPoolExecutor(n, mp_context=ctx,
                initializer=initialize, initargs=(cuda,ctx.Barrier(n))))
            return p,[p.submit(ping) for _ in range(n)]
        gpu,g = pool(count, True)
        cpu,c = pool(2, False)
        comparator,k = pool(1, False)
        for f in g+c+k:
            f.result()
        warm = clock()
        active, pending, queued = {}, {}, {}
        for item in items:
            key = item["id"]
            spec = json.loads(Path(item["manifest"]).read_text())
            queued[key] = clock()
            pending[key] = 2
            for dev in ("cpu","cuda"):
                label = f"{wave}-{dev}"
                uses_gpu = dev == "cuda" or spec["source_device"].split(":")[0] == "cuda"
                p = gpu if uses_gpu else cpu
                f = p.submit(side_task,item,dev,uses_gpu,str(root/key/label))
                active[f] = (item,label)
        while active:
            done, _ = cf.wait(active, return_when=cf.FIRST_COMPLETED)
            for f in done:
                item,label = active.pop(f)
                key = item["id"]
                rec = results.setdefault(key, dict(sides={}, latency=[]))
                if label == "comparison":
                    rec["comparison"] = f.result()
                    rec["latency"].append(clock()-queued[key])
                    continue
                rec["sides"][label] = f.result()
                pending[key] -= 1
                if not pending[key]:
                    if all(v["sig"][0]=="ready" for v in rec["sides"].values()):
                        active[comparator.submit(compare,item,root,wave)] = (item,"comparison")
                    else:
                        rec["comparison"] = signature({})
                        rec["latency"].append(clock()-queued[key])
        end = clock()
    return dict(startup=warm-start,warm=end-warm,teardown=clock()-end)


def trial(root, items, count):
    start = clock()
    results = {}
    waves = [run_wave(root,items,count,0,results)]
    repeat = [i for i in items if i["replay"]]
    if repeat:
        waves.append(run_wave(root,repeat,count,1,results))
    wall = clock()-start
    write(root/"cases.json",results)
    sides = [v for r in results.values() for v in r["sides"].values()]
    lat = sorted(v for r in results.values() for v in r["latency"])
    mismatches = []
    for i in items:
        r = results[i["id"]]
        if i["expected"] != r["comparison"] or i["sides"] != {k:v["sig"] for k,v in r["sides"].items()}:
            mismatches.append(i["id"])
    return dict(workers=count,wall=wall,waves=waves, pairs=len(lat),
                warm=sum(w["warm"] for w in waves),device_bytes=max(v["device_used"] for v in sides),
                latency50=statistics.median(lat),latency95=lat[(95*len(lat)+99)//100-1],
                gpu=next(v["gpu"] for v in sides if v["gpu"]),
                actual_cuda_pids=[len({r["sides"][f"{w}-cuda"]["pid"] for r in results.values()
                                      if f"{w}-cuda" in r["sides"]}) for w in range(len(waves))],
                side_oom=sum(v["sig"][1]=="backend_resource_exhausted" for v in sides),
                archive_disagreements=mismatches)


if __name__ == "__main__":
    root = Path(sys.argv[1])
    items = select()
    write(root/"cohort.json", items)
    summary = []
    for ix,n in enumerate((1,7,10,4,4,10,7,1)):
        dest = root / f"trial-{ix+1}-{n}"
        dest.mkdir()
        for var in ("TRITON_CACHE_DIR","TORCHINDUCTOR_CACHE_DIR"):
            os.environ[var] = str(Path(os.environ[var]).parent / f"{var}-{ix}")
        row = trial(dest,items,n)
        print(ix+1,n,row["wall"],row["warm"],len(row["archive_disagreements"]),flush=True)
        summary.append(row)
        write(root/"summary.json",summary)

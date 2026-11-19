"""Run the fixed 805400 scaling cohort through the production scheduling and replay path."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import sys
import threading
import time

from cuda_worker_scaling import signature, side_signature
from stateful_check_e2e import differential_timing
from mirage.artifacts.replay import differential_members, replay_devices
from mirage.contracts.scheduling import SchedulingOptions
from mirage.runtime.process import run_worker
from mirage.runtime.scheduler import SessionScope, SharedScheduler, session_scope


def read(path):
    return json.loads(path.read_text())


def inventory(archive):
    counts = Counter()
    for session in read(archive/'benchmark-results.json')['sessions']:
        root = Path(session['report']).parent
        old = {(r['attempt'],r['member']) for r in read(root/'differential-ledger.json') if r.get('legal_member')}
        new = {(r['attempt'],m) for r in read(root/'execution/summary.json')['records'] for m in differential_members(r)}
        if old-new:
            raise ValueError(f"{session['api']} lost {len(old-new)} archived legal members")
        counts.update(archived=len(old), additional=len(new-old))
    return counts


def main(archive, cohort_path, output):
    cohort = read(cohort_path)
    counts = inventory(archive)
    scheduler = SharedScheduler(SchedulingOptions(8, 2, 10))
    def replay(item):
        manifest = Path(item['manifest'])
        local = output/'cases'/item['id']/'manifest.json'
        local.parent.mkdir(parents=True)
        shutil.copyfile(manifest,local)
        api = read(local)['api']
        with session_scope(SessionScope(scheduler,api,'cpu',threading.Event())):
            result = replay_devices(local,timeout=120,live=True,admitted=submitted)
        result['legal_member'] = True
        sides = {k:side_signature(v) for k,v in result['devices'].items()}
        failures = []
        if sides != item['sides'] or signature(result.get('comparison') or {}) != item['expected']:
            failures.append('archived_signature')
        workers = {Path(p['directory']).name:p['resources'] for p in result['processes']}
        if item['replay']:
            for device in ('cpu','cuda'):
                repeated = workers.get('1-'+device,{})
                if repeated.get('worker_started',0) < result['timing']['rounds'][0]['compared']:
                    failures.append('replay_worker_not_fresh')
        return result, dict(id=item['id'],failures=failures)

    try:
        submitted = time.monotonic()
        with ThreadPoolExecutor(max_workers=16) as pool:
            records, checks = zip(*pool.map(replay,cohort))
        times = differential_timing(records)
        usage = Counter((p['resources'].get('lane'),p['resources'].get('worker_pid'))
                        for r in records for p in r['processes'])
        cuda_reused = any(lane=='device_cuda' and v>1 for (lane,pid),v in usage.items())
        events = scheduler.snapshot()['events']
        marks = [(e[key], delta) for e in events if e['resource']=='device_cuda' and 'granted' in e
                 for key,delta in [('granted',1),('finished',-1)]]
        active = peak = 0
        for _,delta in sorted(marks):
            active += delta
            peak = max(peak,active)
        request_file = next((Path(p['directory'])/'request.json' for r in records for p in r['processes']
                             if Path(p['directory']).name=='0-comparison'))
        request = read(request_file)
        with session_scope(SessionScope(scheduler,'timeout-control','cpu',threading.Event())):
            timeout = run_worker(request,output/'timeout-control',1e-8)
            recovery = run_worker(request,output/'recovery-control',120)
        control = (timeout.outcome=='timeout' and recovery.outcome=='completed' and
                   timeout.resources['worker_pid']!=recovery.resources['worker_pid'])
        failures = [c for c in checks if c['failures']]
        passed = (not failures and control and cuda_reused and peak==4 and times['missing_timing']==0
                  and times['members_with_both_side_records']==len(cohort)
                  and times['completed_comparisons_including_replays']==len(cohort)+sum(i['replay'] for i in cohort))
        report = dict(passed=passed, inventory=counts, cohort=len(cohort), failures=failures,
                      cuda_peak=peak, cuda_reused=cuda_reused, timeout_recovered=control, timing=times,
                      cohort_sha256=hashlib.sha256(cohort_path.read_bytes()).hexdigest())
        (output/'acceptance.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)
        return 0 if passed else 1
    finally:
        scheduler.differential.close()


if __name__ == '__main__':
    raise SystemExit(main(*(Path(arg) for arg in sys.argv[1:])))

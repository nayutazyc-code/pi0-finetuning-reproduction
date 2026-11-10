"""Transfer disjoint canonical files, then run the unchanged guarded deployment."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = '/root/pi0-repro'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ssh_target')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@-]*', args.ssh_target):
        raise ValueError('Invalid alias')
    out = ROOT / 'outputs/motion_first_20261026'
    plan = json.loads((out / 'training_plan.json').read_text())
    for name, digest in plan['source_sha256'].items():
        assert sha(ROOT / name) == digest, name
    assert sha(ROOT / plan['config']) == plan['config_sha256']
    assert sha(ROOT / plan['dataset'] / 'dataset_manifest.json') == plan['dataset_manifest_sha256']
    repo = ROOT / plan['dataset'] / 'local/pi0_banana_train'
    remote_repo = REMOTE_ROOT + '/' + plan['dataset'] + '/local/pi0_banana_train'
    identity = (ROOT / 'outputs/v6_recovery_20261026/ssh_identity_file.txt').read_text().strip()
    ssh = ['/usr/bin/ssh', '-i', identity, '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes',
           '-o', 'ControlMaster=no', '-o', 'ControlPath=none', '-o', 'ConnectTimeout=12',
           '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3', '-o', 'IPQoS=none']
    code = ("from pathlib import Path;import json;r=Path(" + repr(REMOTE_ROOT) + ");"
            "p=r/'outputs/motion_first_20261026/cloud_training/status.json';"
            "d=Path(" + repr(remote_repo) + ");"
            "print(json.dumps({'status':json.loads(p.read_text()).get('status') if p.exists() else 'not_started',"
            "'files':{str(p.relative_to(d)):p.stat().st_size for p in d.glob('data/chunk-*/episode_*.parquet')}}))")
    remote = REMOTE_ROOT + '/openpi/.venv/bin/python -B -c ' + shlex.quote(code)

    def inventory():
        result = subprocess.run(ssh + [args.ssh_target, remote], capture_output=True, text=True, timeout=25)
        result.check_returncode()
        return json.loads(result.stdout)

    before = inventory()
    if before['status'] != 'not_started':
        print(json.dumps({'existing_job_retained': before['status']}), flush=True)
        return
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S')
    receipt = out / ('parallel_upload_' + stamp)
    receipt.mkdir()
    files = sorted(repo.glob('data/chunk-*/episode_*.parquet'))
    assert len(files) == plan['train_episodes']
    pending = [p for p in files if before['files'].get(str(p.relative_to(repo))) != p.stat().st_size]
    shards, sizes = [[] for _ in range(4)], [0] * 4
    for path in sorted(pending, key=lambda p: p.stat().st_size, reverse=True):
        index = sizes.index(min(sizes))
        shards[index].append(str(path.relative_to(repo)))
        sizes[index] += path.stat().st_size
    flattened = [name for group in shards for name in group]
    assert len(flattened) == len(set(flattened)) == len(pending)
    for index, group in enumerate(shards):
        (receipt / f'shard_{index}.txt').write_text('\n'.join(group) + ('\n' if group else ''))
    record = {'pid': os.getpid(), 'status': 'transferring', 'shards': shards, 'shard_bytes': sizes,
              'pending_files': len(pending), 'remaining_bytes': sum(sizes),
              'training_plan_sha256': sha(out / 'training_plan.json'),
              'scope': 'Disjoint canonical files only. Size inventory selects transfer candidates; '
                       'the unchanged final deployment rechecks checksums and metadata before launch.'}
    (receipt / 'plan.json').write_text(json.dumps(record, indent=2) + '\n')
    (out / 'parallel_upload_current.json').write_text(json.dumps({'pid': os.getpid(), 'receipt': str(receipt)}) + '\n')
    print(json.dumps({key: record[key] for key in ('pid', 'pending_files', 'remaining_bytes', 'shard_bytes')}), flush=True)

    def interrupted(signum, frame):
        raise InterruptedError('Parallel upload interrupted')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    inhibitor = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(os.getpid())])
    children = []
    try:
        remaining = [index for index, group in enumerate(shards) if group]
        for attempt in range(1, 4):
            if inventory()['status'] != 'not_started':
                raise RuntimeError('A supervisor appeared; refusing concurrent data mutation')
            children = []
            for index in remaining:
                log = (receipt / f'shard_{index}_attempt_{attempt}.log').open('xb')
                command = ['rsync', '-rlpt', '--checksum', '--partial',
                           '--files-from=' + str(receipt / f'shard_{index}.txt'), '-e', shlex.join(ssh),
                           str(repo) + '/', args.ssh_target + ':' + remote_repo + '/']
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                log.close()
                children.append((index, child))
            results = {index: child.wait() for index, child in children}
            print(json.dumps({'round': attempt, 'transfer_returncodes': results}), flush=True)
            remaining = [index for index, result in results.items() if result]
            if not remaining:
                break
            if any(results[index] not in (12, 20, 23, 255) for index in remaining) or attempt == 3:
                raise RuntimeError('Transfer failure retained; see per-shard logs')
            time.sleep(5)
        record['status'] = 'transferred_pending_final_checksums'
        (receipt / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
        child = subprocess.Popen([str(ROOT / '.venv/bin/python'), '-B',
                                  str(ROOT / 'scripts/run_motion_deployment_with_key.py'), args.ssh_target],
                                 cwd=ROOT, start_new_session=True)
        children = [(-1, child)]
        result = child.wait()
        if result:
            raise RuntimeError('Final guarded deployment failed: ' + str(result))
        record['status'] = 'guarded_deployment_returned'
        (receipt / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
    finally:
        for _, child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        inhibitor.terminate()


if __name__ == '__main__':
    main()

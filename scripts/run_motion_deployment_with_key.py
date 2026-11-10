"""Retry only transport failures; preserve the frozen training plan and jobs."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = '/root/pi0-repro'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ssh_target')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@-]*', args.ssh_target):
        raise ValueError('Invalid SSH alias')
    identity = Path((ROOT / 'outputs/v6_recovery_20261026/ssh_identity_file.txt').read_text().strip())
    if not identity.is_file() or identity.is_symlink():
        raise ValueError('Dedicated identity is unavailable')
    ssh = ['/usr/bin/ssh', '-i', str(identity), '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes',
           '-o', 'ConnectTimeout=12', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3']
    code = ("from pathlib import Path;import json;"
            "p=Path('/root/pi0-repro/outputs/motion_first_20261026/cloud_training/status.json');"
            "print(json.dumps({'status':json.loads(p.read_text()).get('status')} if p.exists() else {'status':'not_started'}))")
    remote = REMOTE_ROOT + '/openpi/.venv/bin/python -B -c ' + shlex.quote(code)

    def status():
        result = subprocess.run(ssh + ['-o', 'ControlMaster=no', '-o', 'ControlPath=none',
                                      args.ssh_target, remote], capture_output=True, text=True, timeout=25)
        result.check_returncode()
        return json.loads(result.stdout)['status']

    def bootstrap_base_lineage():
        # Older deployments sent only the training repository. The new guard
        # compares held-out lineage, so send its two small mappings as well.
        plan = json.loads((ROOT / 'outputs/motion_first_20261026/training_plan.json').read_text())
        base = Path(plan['base_dataset'])
        if base.is_absolute() or '..' in base.parts:
            raise ValueError('Invalid base dataset path')
        mappings = [base / f'local/pi0_banana_{split}/source_episodes.json'
                    for split in ('validation', 'test')]
        expected = {}
        import hashlib
        for relative, split in zip(mappings, ('validation', 'test'), strict=True):
            path = ROOT / relative
            record = json.loads(path.read_text())
            assert record['source_split'] == split and all(row['split'] == split for row in record['episodes'])
            expected[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
        check = ("from pathlib import Path;import hashlib,json; r=Path(" + repr(REMOTE_ROOT) + ");"
                 "expected=" + repr(expected) + ";"
                 "assert all(not(r/p).exists() or hashlib.sha256((r/p).read_bytes()).hexdigest()==h "
                 "for p,h in expected.items()), 'Existing base lineage differs';"
                 "print('Base held-out mapping compatibility checked')")
        result = subprocess.run(ssh + ['-o', 'ControlMaster=no', '-o', 'ControlPath=none',
                                      args.ssh_target, REMOTE_ROOT + '/openpi/.venv/bin/python -B -c ' + shlex.quote(check)],
                                capture_output=True, text=True, timeout=25)
        result.check_returncode()
        listing = ROOT / 'outputs/motion_first_20261026/base_lineage_metadata_files.txt'
        listing.write_text('\n'.join(expected) + '\n')
        subprocess.run(['rsync', '-rlpt', '--checksum', '--relative', '--files-from=' + str(listing),
                        '-e', shlex.join(ssh + ['-o', 'ControlMaster=no', '-o', 'ControlPath=none']),
                        str(ROOT) + '/', args.ssh_target + ':' + REMOTE_ROOT + '/'], check=True)
        print('Base held-out lineage mappings supplied; no held-out images/actions transferred.', flush=True)

    def interrupted(signum, frame):
        raise InterruptedError('Deployment wrapper interrupted')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    # The transport shim supplies only identity/liveness options. Bound source,
    # config, sampler and dataset bytes are not edited to implement a retry.
    with tempfile.TemporaryDirectory(prefix='pi0-motion-transport-', dir='/private/tmp') as temporary:
        shim = Path(temporary) / 'ssh'
        shim.write_text('#!/bin/sh\nexec ' + shlex.join(ssh) + ' "$@"\n')
        shim.chmod(0o700)
        env = dict(os.environ, PATH=temporary + os.pathsep + os.environ['PATH'])
        for attempt in range(1, 5):
            existing = status()  # Unknown state is never permission to relaunch.
            if existing != 'not_started':
                print(json.dumps({'status': 'existing_job_retained', 'remote_status': existing}), flush=True)
                return
            bootstrap_base_lineage()
            print(json.dumps({'deployment_attempt': attempt, 'remote_status': existing}), flush=True)
            child = subprocess.Popen(['bash', str(ROOT / 'scripts/deploy_motion_training.sh'), args.ssh_target],
                                     cwd=ROOT, env=env, start_new_session=True)
            try:
                returncode = child.wait()
            except BaseException:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                raise
            if returncode == 0:
                print(json.dumps({'deployment': 'dispatch_returned', 'remote_status': status()}), flush=True)
                return
            if returncode not in (12, 20, 23, 255):
                raise RuntimeError('Non-transport deployment failure: ' + str(returncode))
            existing = status()
            if existing != 'not_started':
                print(json.dumps({'status': 'existing_job_retained', 'remote_status': existing}), flush=True)
                return
            if attempt == 4:
                raise RuntimeError('Four transport attempts exhausted; uploaded files retained')
            print(json.dumps({'retry_transport_exit': returncode, 'job_not_started_verified': True}), flush=True)
            time.sleep(5)


if __name__ == '__main__':
    main()

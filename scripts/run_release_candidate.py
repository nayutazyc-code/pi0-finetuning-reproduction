"""Run the opt-in release/retreat candidate and restore v6 on exit."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'repos/openpi/packages/openpi-client/src')]
from pi0_repro.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Validate local configuration; no SSH, inference or simulation')
    parser.add_argument('--display', action='store_true', help='Show the native viewer; display overhead was not part of the headless comparison')
    args = parser.parse_args()
    cfg = load_config(ROOT / 'configs/banana_ik_continuous_damping2_pi0.toml')
    assert cfg.model.expected_checkpoint_suffix == 'banana_v6_ik_candidate1000/999'
    from pi0_repro.scene import load_scene_config
    scene = load_scene_config(Path(cfg.simulation.scene_config))
    assert scene.wrist_damping_scale == 2 and scene.controller_profile == 'position_pd'
    assert cfg.simulation.rtc_blend_steps == cfg.simulation.chunk_blend_steps == 20
    evidence = json.loads((ROOT / 'outputs/release_retreat_20261029/closed_loop/audit.json').read_text())
    assert evidence['passed'], 'Candidate needs the completed audited pilot'
    if args.check_only:
        print('Local configuration verified. Experimental candidate; default unchanged. No remote action.')
        return

    target = (ROOT / 'outputs/random_multiview_connection/ssh_target.txt').read_text().strip()
    assert re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@-]*', target), 'Invalid SSH target'
    raw = subprocess.check_output(['route', '-n', 'get', target], text=True, timeout=15)
    interface = re.search(r'interface:\s*(\S+)', raw)
    assert interface and not interface.group(1).startswith(('utun', 'tun', 'tap')), 'Disable TUN before running the cloud comparison'
    ssh = ['ssh', '-i', str(Path.home() / '.ssh/id_ed25519'), '-o', 'IdentitiesOnly=yes',
           '-o', 'BatchMode=yes', '-o', 'ControlMaster=no', '-o', 'ControlPath=none',
           '-o', 'ConnectTimeout=10', '-o', 'IPQoS=none', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3']
    remote = '/root/pi0-repro'

    def select(model):
        command = [remote + '/openpi/.venv/bin/python', '-u', remote + '/outputs/ik_continuous_20261029/cloud_policy_select.py',
                   '--root', remote, '--model', model]
        result = subprocess.run(ssh + [target, shlex.join(command)], capture_output=True, text=True, timeout=60)
        result.check_returncode()
        return json.loads(result.stdout)

    from pi0_repro.remote import RemotePi0

    def endpoint(config):
        deadline = time.monotonic() + 100
        while True:
            try:
                with RemotePi0(config) as client:
                    return client.metadata
            except (OSError, TimeoutError):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(1)
            except RuntimeError as exc:
                if 'ConnectionRefusedError' not in str(exc) or time.monotonic() >= deadline:
                    raise
                time.sleep(1)

    cfg.connection.port = 18083
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', cfg.connection.port))
    tunnel = subprocess.Popen(ssh + ['-o', 'ExitOnForwardFailure=yes', '-N', '-L',
                                     '127.0.0.1:18083:127.0.0.1:8001', target])
    changed = False
    try:
        # Mark before dispatch: even an interrupted SSH response may have switched the server.
        changed = True
        select('candidate')
        print('Candidate metadata verified:', endpoint(cfg)['checkpoint'], flush=True)
        os.environ.setdefault('MUJOCO_GL', 'glfw')
        from pi0_repro.simulation import run
        print('Saved run:', run(cfg, display=args.display), flush=True)
    finally:
        try:
            if changed:
                select('v6')
                rollback = load_config(ROOT / 'configs/banana_v6_bridge20_pi0.toml')
                rollback.connection.port = cfg.connection.port
                print('Restored v6:', endpoint(rollback)['checkpoint'], flush=True)
        finally:
            tunnel.terminate()
            try:
                tunnel.wait(timeout=5)
            except subprocess.TimeoutExpired:
                tunnel.kill()
                tunnel.wait()


if __name__ == '__main__':
    main()

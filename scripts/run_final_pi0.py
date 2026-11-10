"""User-selected candidate workflow; preserve selection on success, fall back on error."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
import json
import logging
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'repos/openpi/packages/openpi-client/src')]
from pi0_repro.config import load_config
from pi0_repro.scene import load_scene_config
from pi0_repro.remote import RemotePi0

CANDIDATE='banana_v6_ik_candidate1000/999'
V6='banana_v6_long100_accum8000/7999'
NORM='6c6e7a92643c76996bf15ba9fc3d3e346dd6bc3d8697712118d9fdb49dedf269'
REMOTE='/root/pi0-repro'

def local_config(fallback=False):
    if fallback:
        cfg=load_config(ROOT/'configs/banana_v6_bridge20_pi0.toml')
    else:
        cfg=load_config(ROOT/'configs/banana_final_pi0.toml')
        original=load_config(ROOT/'configs/banana_ik_continuous_damping2_pi0.toml')
        assert asdict(cfg)==asdict(original),'Final profile differs from selected candidate settings'
    scene=load_scene_config(Path(cfg.simulation.scene_config))
    assert cfg.model.expected_checkpoint_suffix==(V6 if fallback else CANDIDATE)
    assert cfg.model.expected_norm_stats_sha256==NORM
    assert scene.wrist_damping_scale==(1 if fallback else 2) and scene.controller_profile=='position_pd'
    assert cfg.model.send_full_resolution and cfg.simulation.replan_steps==100
    assert cfg.simulation.rtc_enabled and cfg.simulation.rtc_blend_steps==cfg.simulation.chunk_blend_steps==20
    assert cfg.simulation.joint_target_filter_hz==3
    return cfg,scene

class Session:
    """Own only this invocation's forwarding and local receipts."""
    def __init__(self,cfg,scene,*,fallback=False,select_only=False,display=False):
        self.cfg,self.scene=cfg,scene
        self.fallback,self.select_only,self.display=fallback,select_only,display
        self.target=None;self.ssh=None;self.out=None;self.tunnel=None;self.tunnel_log=None
        self.handler=None;self.previous_log_level=None;self.changed=False;self.record={}

    def clean(self,message):
        return str(message).replace(self.target,'[saved target]') if self.target else str(message)

    def save(self,name,data):
        p=self.out/name;q=p.with_suffix('.tmp')
        q.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n');q.replace(p)

    def setup(self):
        stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]
        self.out=ROOT/'outputs'/('final_runtime_'+stamp);self.out.mkdir()
        self.target=(ROOT/'outputs/random_multiview_connection/ssh_target.txt').read_text().strip()
        if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@-]*',self.target):raise ValueError('Invalid saved SSH target')
        self.ssh=['ssh','-i',str(Path.home()/'.ssh/id_ed25519'),'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ConnectTimeout=10','-o','IPQoS=none','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3']
        raw=subprocess.check_output(['route','-n','get',self.target],text=True,timeout=15)
        route=re.search(r'interface:\s*(\S+)',raw)
        if not route:raise RuntimeError('Network interface unavailable')
        self.record=dict(success=False,selected_model='v6' if self.fallback else 'candidate',route_interface=route.group(1),select_only=self.select_only,fallback_v6=self.fallback)
        self.save('plan.json',dict(self.record,scope='User-selected runtime, not a new acceptance experiment; no training or weight changes',on_success='Keep selected model',on_error='Attempt verified v6 fallback'))
        self.save('config.json',dict(config=asdict(self.cfg),scene=asdict(self.scene)))
        self.handler=logging.FileHandler(self.out/'runtime.log');self.handler.setLevel(logging.INFO)
        self.previous_log_level=logging.getLogger().level
        logging.getLogger().setLevel(logging.INFO);logging.getLogger().addHandler(self.handler)
        print('Runtime receipts:',self.out,flush=True)

    def remote(self,args,timeout=75):
        result=subprocess.run(self.ssh+[self.target,shlex.join(args)],capture_output=True,text=True,timeout=timeout)
        if result.returncode:raise RuntimeError('Remote command failed: '+self.clean(result.stderr[-1800:]))
        return json.loads(result.stdout)

    def ensure_proxy(self):
        # A listening port is preserved. Subsequent metadata validation decides
        # compatibility; an unfamiliar process is never killed or overwritten.
        code="""from pathlib import Path
import json,socket,subprocess,sys,time
root=Path('/root/pi0-repro')
s=socket.socket();s.settimeout(2);listening=s.connect_ex(('127.0.0.1',8001))==0;s.close()
receipt={'port':8001,'existing':listening,'started':False}
if not listening:
 out=root/'outputs/final_runtime'/sys.argv[1];out.mkdir(parents=True,exist_ok=False)
 with (out/'proxy.log').open('ab') as log:
  p=subprocess.Popen([sys.executable,'-u',str(root/'server/image_transport_proxy.py'),'--host','127.0.0.1','--port','8001','--upstream-port','8000'],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 time.sleep(.3)
 if p.poll() is not None:raise RuntimeError('Owned PNG proxy exited; inspect its retained log')
 receipt.update(started=True,pid=p.pid)
print(json.dumps(receipt))
"""
        receipt=self.remote([REMOTE+'/openpi/.venv/bin/python','-B','-c',code,self.out.name],timeout=30)
        self.save('proxy.json',receipt)

    def start_tunnel(self):
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        self.cfg.connection.port=port;self.cfg.connection.connect_timeout_seconds=5
        self.tunnel_log=(self.out/'tunnel.log').open('x')
        self.tunnel_log.write('Owned SSH forwarding; remote target omitted. Endpoint metadata verifies connectivity.\n');self.tunnel_log.flush()
        self.tunnel=subprocess.Popen(self.ssh+['-o','ExitOnForwardFailure=yes','-N','-L',f'127.0.0.1:{port}:127.0.0.1:8001',self.target],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

    def select(self,model):
        self.changed=True  # A lost dispatch response can still have switched policy.
        receipt=self.remote([REMOTE+'/openpi/.venv/bin/python','-u',REMOTE+'/outputs/ik_continuous_20261029/cloud_policy_select.py','--root',REMOTE,'--model',model])
        self.save(model+'_dispatch.json',receipt)
        return receipt

    def endpoint(self,cfg):
        deadline=time.monotonic()+360
        while True:
            try:
                with RemotePi0(cfg) as client:
                    m=client.metadata
                    if m.get('denoise_steps')!=10:raise ValueError('Unexpected denoise count')
                    return m
            except (OSError,TimeoutError):
                if time.monotonic()>=deadline:raise
                time.sleep(1)
            except RuntimeError as e:
                if not any(s in str(e) for s in ('ConnectionRefusedError','connection refused')) or time.monotonic()>=deadline:raise
                time.sleep(1)

    def simulate(self):
        os.environ.setdefault('MUJOCO_GL','glfw')
        from pi0_repro.simulation import run
        path=run(self.cfg,display=self.display)
        self.record['run_directory']=str(path)
        summary=json.loads((Path(path)/'summary.json').read_text())
        self.record['episodes']=[dict(success=r['success'],steps=r['steps']) for r in summary['episodes']]
        if any(r.get('aborted_by_window_close') or r.get('error') for r in summary['episodes']):
            raise RuntimeError('Simulation aborted; restoring v6')
        return path

    def restore(self):
        self.select('v6')
        cfg,_=local_config(True)
        cfg.connection.port=self.cfg.connection.port;cfg.connection.connect_timeout_seconds=5
        metadata=self.endpoint(cfg);self.save('fallback_v6_metadata.json',metadata)
        self.record['fallback_verified']=True

    def close(self):
        if self.tunnel is not None:
            self.tunnel.terminate()
            try:self.tunnel.wait(timeout=5)
            except subprocess.TimeoutExpired:self.tunnel.kill();self.tunnel.wait()
        if self.tunnel_log is not None:self.tunnel_log.close()
        if self.handler is not None:logging.getLogger().removeHandler(self.handler);self.handler.close()
        if self.previous_log_level is not None:logging.getLogger().setLevel(self.previous_log_level)

    def execute(self):
        try:
            self.setup();self.ensure_proxy();self.start_tunnel()
            model='v6' if self.fallback else 'candidate'
            self.select(model);metadata=self.endpoint(self.cfg);self.save('selected_metadata.json',metadata)
            if not self.select_only and not self.fallback:
                path=self.simulate();print('Saved run:',path,flush=True)
            self.record.update(success=True,selected_metadata_verified=True,kept_selected_service=True)
            print('Selected service retained:',metadata['checkpoint'],flush=True)
        except BaseException as error:
            self.record.update(success=False,error_type=type(error).__name__,error=self.clean(error))
            if self.changed:
                try:self.restore()
                except BaseException as fallback_error:
                    self.record['fallback_verified']=False
                    self.record['fallback_error']=self.clean(fallback_error)
                    print('v6 fallback failed; original error follows:',self.clean(fallback_error),file=sys.stderr,flush=True)
            raise
        finally:
            try:
                self.close()
            except BaseException as close_error:
                self.record['cleanup_error']=self.clean(close_error)
                print('Owned tunnel cleanup error:',self.clean(close_error),file=sys.stderr)
            finally:
                if self.out is not None:
                    try:self.save('completion.json',self.record)
                    except BaseException as receipt_error:
                        print('Completion receipt could not be saved:',self.clean(receipt_error),file=sys.stderr)
                        if 'error' not in self.record:raise
        return self.record

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only',action='store_true',help='Validate local configuration only; no writes, SSH or model calls')
    mode.add_argument('--select-only',action='store_true',help='Select and verify candidate service without stepping simulation')
    mode.add_argument('--fallback-v6',action='store_true',help='Select and verify v6 without stepping simulation')
    parser.add_argument('--display',action='store_true')
    parser.add_argument('--episodes',type=int,default=1)
    parser.add_argument('--seed',type=int)
    args=parser.parse_args(argv)
    if args.episodes<1:parser.error('--episodes must be positive')
    cfg,scene=local_config(args.fallback_v6)
    if args.check_only:
        print(json.dumps(dict(valid=True,config=asdict(cfg),scene=asdict(scene)),indent=2));return
    cfg.simulation.episodes=args.episodes;cfg.simulation.display=args.display
    if args.seed is not None:cfg.simulation.seed=args.seed
    Session(cfg,scene,fallback=args.fallback_v6,select_only=args.select_only,display=args.display).execute()

if __name__=='__main__':main()

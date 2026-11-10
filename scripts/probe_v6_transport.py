"""Paired synthetic SSH transfer and no-step pi0 requests with proxy spans.

Uses an owned tunnel and the proxy's bounded diagnostic endpoint. No historical latency attribution,
environment control, model change, or training data collection.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'repos/openpi/packages/openpi-client/src')]
os.environ.setdefault('MUJOCO_GL','glfw')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--pairs',type=int,default=40)
    args=p.parse_args();out=args.out.resolve();assert out.is_relative_to(ROOT/'outputs') and 1<=args.pairs<=100
    out.mkdir(parents=True,exist_ok=False)
    import numpy as np
    from pi0_repro.config import load_config
    from pi0_repro.remote import RemotePi0
    from pi0_repro.simulation import _make_env
    from pi0_repro.observations import aloha_observation
    from profile_v6_runtime import distribution,sha
    from websockets.sync.client import connect
    from openpi_client import msgpack_numpy
    target=(ROOT/'outputs/random_multiview_connection/ssh_target.txt').read_text().strip()
    import re
    assert re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@-]*',target)
    plan=dict(schema='v6-transport-span-probe-v1',pairs=args.pairs,synthetic_request_bytes=55000,synthetic_response_bytes=8192,
              script_sha256=sha(Path(__file__)),source_proxy_sha256=sha(ROOT/'server/image_transport_proxy.py'),
              scope='New same-SSH-connection bounded WebSocket echo and sequential no-environment-step inference through allowed8001. Synthetic echo is a path diagnostic, not exact policy workload or proof of a historical network cause. Alternate pair order; retain all warmups/outliers.')
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');(out/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    report=dict(plan=plan,complete=False,warmups=[],pairs=[])
    def save():(out/'probe.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    cfg=load_config(ROOT/'configs/banana_v6_bridge20_pi0.toml');cfg.connection.port=18063;cfg.simulation.display=False;cfg.simulation.save_video=False
    env=_make_env(cfg,out);obs,_=env.reset(seed=cfg.simulation.seed)
    observation=aloha_observation({'pixels':{'top':env.render()},'agent_pos':obs['agent_pos']},cfg.simulation.prompt,resize=False)
    prefix=np.repeat(obs['agent_pos'][None],100,axis=0);prefix[:,[6,13]]=np.clip(prefix[:,[6,13]],0,1)
    observation['rtc']=dict(schema='pi0-rtc-v1',prefix=prefix,valid_steps=100,delay_steps=10,blend_steps=20,guidance_max=.5)
    env.close()
    ssh=['ssh','-N','-i',str(Path.home()/'.ssh/id_ed25519'),'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ConnectTimeout=10','-o','IPQoS=none','-o','Compression=no','-o','ExitOnForwardFailure=yes',
         '-L','127.0.0.1:18063:127.0.0.1:8001',target]
    log=(out/'ssh.log').open('xb');process=None;echo=None
    save()
    try:
        for port in (18063,):
            with socket.socket() as s:s.bind(('127.0.0.1',port))
        process=subprocess.Popen(ssh,stdout=log,stderr=log)
        deadline=time.monotonic()+20
        while True:
            if process.poll() is not None:raise RuntimeError('Owned tunnel exited')
            try:
                echo=connect('ws://127.0.0.1:18063/transport-probe',compression=None,max_size=1024*1024,open_timeout=5)
                break
            except OSError:
                if time.monotonic()>deadline:raise
                time.sleep(.2)
        assert msgpack_numpy.unpackb(echo.recv(timeout=10))['schema']=='pi0-transport-echo-v1'
        payload=np.random.default_rng(20261028).bytes(55000)
        def transfer():
            start=time.perf_counter();echo.send(payload)
            assert echo.recv(timeout=15)==payload[:8192]
            return (time.perf_counter()-start)*1000
        with RemotePi0(cfg) as client:
            report['metadata']=client.metadata
            assert client.metadata['transport_proxy'].get('timing_schema')=='pi0-proxy-timing-v1'
            def infer():
                actions,ms,result=client.infer(observation)
                assert actions.shape==(100,14) and np.isfinite(actions).all()
                t=result['transport_timing'];assert t['pixel_digest_verified'] and t['timing_schema']=='pi0-proxy-timing-v1'
                enc=result['client_transport']['encoding_ms'];proxy=t['proxy_handler_ms_before_reply_pack']
                return dict(round_trip_ms=ms,client_transport=result['client_transport'],transport_timing=t,
                            server_timing=result.get('server_timing'),policy_timing=result.get('policy_timing'),
                            outside_encoding_and_proxy_handler_ms=ms-enc-proxy)
            for i in range(6):report['warmups'].append(dict(index=i,echo_ms=transfer(),inference=infer()));save()
            for i in range(args.pairs):
                row=dict(index=i,order='echo_then_model' if i%2==0 else 'model_then_echo')
                if i%2==0:row.update(echo_ms=transfer(),inference=infer())
                else:row.update(inference=infer(),echo_ms=transfer())
                report['pairs'].append(row);save()
                print('PAIR',i+1,'echo',round(row['echo_ms'],2),'model',round(row['inference']['round_trip_ms'],2),flush=True)
        report['summary']=dict(echo_ms=distribution([r['echo_ms'] for r in report['pairs']]),
            model_round_trip_ms=distribution([r['inference']['round_trip_ms'] for r in report['pairs']]),
            proxy_handler_ms=distribution([r['inference']['transport_timing']['proxy_handler_ms_before_reply_pack'] for r in report['pairs']]),
            proxy_upstream_round_trip_ms=distribution([r['inference']['transport_timing']['proxy_upstream_round_trip_ms'] for r in report['pairs']]),
            outside_encoding_and_proxy_ms=distribution([r['inference']['outside_encoding_and_proxy_handler_ms'] for r in report['pairs']]))
        report['complete']=True
    except BaseException as exc:report['error']=f'{type(exc).__name__}: {exc}';raise
    finally:
        save()
        if echo is not None:
            echo.close()
        if process is not None:
            process.terminate()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=5)
        log.close()
    print(json.dumps(report['summary'],indent=2),flush=True)


if __name__=='__main__':main()

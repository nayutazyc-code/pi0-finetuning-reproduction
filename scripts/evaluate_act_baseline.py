"""Closed-loop ACT data-learnability check with truthful ACT metadata."""
import argparse,hashlib,io,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'repos/openpi/packages/openpi-client/src')]
import numpy as np
from PIL import Image
from openpi_client import msgpack_numpy
from websockets.sync.client import connect

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')

def main():
 p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');p.add_argument('--port',type=int,default=18084);args=p.parse_args();out=ROOT/'outputs/act_baseline_20261029';base=ROOT/'data/lerobot/20261029-135803-ik-replaced-b8ed3a';manifest=json.loads((base/'dataset_manifest.json').read_text());raw=Path(manifest['source_path']);cases=[]
 for split,repeats,limit in [('train',1,4),('validation',2,6)]:
  mapping=json.loads((base/f'local/pi0_banana_{split}/source_episodes.json').read_text())
  for rep in range(repeats):
   for r in mapping['episodes'][:limit]:
    d=json.loads((raw/r['source_episode']/'layout.json').read_text());scene=dict(d['scene']);scene['wrist_damping_scale']=1.;cases.append(dict(index=len(cases),split=split,repeat=rep+1,layout=r['source_episode'],seed=d['reset_seed'],scene=scene))
 plan=dict(cases=cases,attempts=16,model_family='act',checkpoint_step=20000,chunk=100,execute_steps=25,temporal_ensemble=False,rtc=False,max_steps=2000,control_dt=.02,real_time=False,plant='position_pd damping1',limits='Environment actuator bounds only; no pi0 target filter/RTC/blend. Synchronous cloud queries incur wall waits, not a realtime timing benchmark.',selection='Fixed final20000 checkpoint, no outcome-based rerun or checkpoint choice;4training memorization checks and12validation attempts kept separate; no test/OOD tuning.',source_sha256={n:sha(ROOT/'src/pi0_repro'/n) for n in ['banana_env.py','scene.py','random_pose.py']},runner_sha256=sha(Path(__file__)))
 ep=out/'evaluation_plan.json'
 if ep.exists():assert json.loads(ep.read_text())==plan
 else:write(ep,plan)
 if args.prepare_only:print('PREPARED16');return
 os.environ.setdefault('MUJOCO_GL','glfw')
 import imageio.v2 as imageio
 from pi0_repro.banana_env import BananaBoxEnv
 from pi0_repro.scene import SceneConfig
 pack=msgpack_numpy.Packer();ledger=dict(complete=False,plan=plan,results=[]);lp=out/'evaluation.json';assert not lp.exists();write(lp,ledger)
 for case in cases:
  path=out/f"evaluation_{case['index']:02d}";path.mkdir();row=dict(case,run_directory=str(path));ledger['results'].append(row);write(lp,ledger);env=None;ws=None;writer=None;states=[];actions=[];requested=[];rtts=[];task=[]
  try:
   assert all(sha(ROOT/'src/pi0_repro'/n)==h for n,h in plan['source_sha256'].items())
   env=BananaBoxEnv(SceneConfig(**case['scene']),max_episode_steps=2000,render_observations=False);obs,_=env.reset(seed=case['seed']);(path/'scene.xml').write_text(env.xml);write(path/'reset.json',env.last_reset);ws=connect(f'ws://127.0.0.1:{args.port}',compression=None,max_size=16*1024*1024,open_timeout=15);meta=msgpack_numpy.unpackb(ws.recv(timeout=30));assert meta['model_family']=='act' and meta['step']==20000 and meta['action_dim']==14 and meta['action_horizon']==100 and meta['action_semantics']=='native_aloha_absolute';write(path/'server.json',meta)
   if ledger.get('checkpoint_sha256') is None:ledger['checkpoint_sha256']=meta['checkpoint_sha256']
   assert ledger['checkpoint_sha256']==meta['checkpoint_sha256'];writer=imageio.get_writer(path/'episode.mp4',fps=25);start=time.perf_counter();chunk=None
   for step in range(2000):
    if step%25==0:
     frame=env.render();buf=io.BytesIO();Image.fromarray(frame).save(buf,format='PNG',compress_level=1);arr=frame.transpose(2,0,1).copy();message=dict(state=obs['agent_pos'],images_png={'cam_high':buf.getvalue()},image_sha256={'cam_high':hashlib.sha256(arr.tobytes()).hexdigest()});tic=time.perf_counter();ws.send(pack.pack(message));reply=msgpack_numpy.unpackb(ws.recv(timeout=120));rtts.append((time.perf_counter()-tic)*1000);chunk=np.asarray(reply['actions'],np.float32);assert chunk.shape==(100,14) and np.isfinite(chunk).all();assert reply['transport']['pixel_digest_verified']
    if step%2==0:writer.append_data(env.render())
    states.append(obs['agent_pos'].copy());action=chunk[step%25].copy();requested.append(action);obs,reward,terminated,truncated,info=env.step(action);actions.append(info.pop('executed_action'));task.append(info)
    if terminated or truncated:break
   row.update(success=bool(info['is_success']),steps=len(states),simulated_seconds=len(states)*.02,wall_seconds=time.perf_counter()-start,ever_grasped=info['ever_grasped'],ever_lifted=info['ever_lifted'],inference_calls=len(rtts),rtt_ms=dict(median=float(np.median(rtts)),max=max(rtts)));writer.append_data(env.render());np.savez_compressed(path/'trajectory.npz',state=states,action=actions,requested_action=requested,rtt_ms=rtts);(path/'task.jsonl').write_text('\n'.join(json.dumps(r) for r in task)+'\n');print('END',case['index'],case['split'],row['success'],flush=True)
  except BaseException as exc:row['error']=repr(exc);raise
  finally:
   if writer is not None:writer.close()
   if ws is not None:ws.close()
   if env is not None:env.close()
   write(lp,ledger)
 ledger['complete']=True;ledger['groups']={s:dict(successes=sum(r['success'] for r in ledger['results'] if r['split']==s),attempts=sum(r['split']==s for r in ledger['results'])) for s in ['train','validation']};write(lp,ledger);print('COMPLETE',ledger['groups'],flush=True)
if __name__=='__main__':main()

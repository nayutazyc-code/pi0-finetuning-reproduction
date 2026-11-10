"""Physical old/new expert screen on fixed training layouts; no policy inference."""
from pathlib import Path
import argparse
import json
import hashlib
import os
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));os.environ.setdefault('MUJOCO_GL','glfw')
from pi0_repro.banana_env import BananaBoxEnv
from pi0_repro.banana_expert import ScriptedBananaExpert,load_expert_config
from pi0_repro.motion_quality import motion_quality
from pi0_repro.scene import SceneConfig


class Finished(Exception):pass


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    source=ROOT/'data/banana_demos/20261026-142111-merged-e54349'
    cases=['episode_000','episode_002','episode_024']
    report={'scope':'Fixed three train layouts, one physical rollout per expert. No policy/task-quality generalization, no source data modified.',
            'cases':cases,'expert_source_sha256':hashlib.sha256((ROOT/'src/pi0_repro/banana_expert.py').read_bytes()).hexdigest(),
            'profile_sha256':{name:hashlib.sha256((ROOT/'configs'/cfg).read_bytes()).hexdigest() for name,cfg in
                              (('legacy','banana_quality_expert.toml'),('motion','banana_motion_expert.toml'))},
            'complete':False,'results':[]}
    for name in cases:
        meta=json.loads((source/name/'episode.json').read_text());assert meta['split']=='train'
        for profile,cfg in (('legacy','banana_quality_expert.toml'),('motion','banana_motion_expert.toml')):
            config=load_expert_config(ROOT/'configs'/cfg)
            directory=out/f'{name}_{profile}';directory.mkdir()
            env=BananaBoxEnv(SceneConfig(**meta['scene']),max_episode_steps=config.max_steps,render_observations=False)
            obs,info=env.reset(seed=meta['reset_seed']);states=[];actions=[];phases=[];tasks=[];lost=0;expert=None;error=None
            def execute(action,phase,target):
                nonlocal obs,info,lost
                before=obs['agent_pos'].copy();obs,reward,terminated,truncated,info=env.step(action)
                states.append(before);actions.append(info['executed_action']);phases.append(phase)
                tasks.append({k:v for k,v in info.items() if k!='executed_action'})
                lost=lost+1 if phase in ('transport','lower_into_box') and info['released'] else 0
                if lost>=3:raise RuntimeError('Lost controlled contact before planned release')
                if terminated:
                    if phase not in ('release','retreat','settle'):raise RuntimeError('Unintentional release')
                    raise Finished
                if truncated:raise RuntimeError('Step limit')
            try:
                expert=ScriptedBananaExpert(env,config,execute);expert.run()
            except Finished:pass
            except BaseException as exc:error=f'{type(exc).__name__}: {exc}'
            finally:
                if expert is not None:expert.close()
                Image.fromarray(env.physics.render(height=480,width=640,camera_id='angle')).save(directory/'terminal_angle.png')
                env.close()
            np.savez_compressed(directory/'trajectory.npz',state=np.asarray(states),action=np.asarray(actions),phase=np.asarray(phases))
            row={'layout':name,'profile':profile,'steps':len(states),'success':bool(info.get('is_success')),'error':error,
                 'motion_quality':motion_quality(states,actions,tasks) if len(states)>=4 else None}
            (directory/'task.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in tasks))
            report['results'].append(row);(out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
            metric=row['motion_quality'];j=metric['active_frame_max_jerk_rad_s3'] if metric else None
            print(profile,name,'success',row['success'],'activejerkp99',j['p99'] if j else None,'error',error,flush=True)
    report['complete']=True;(out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

"""Acquire learner states only on the newly preassigned training layouts."""
from pathlib import Path
import json
import os
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'repos/openpi/packages/openpi-client/src')]
os.environ.setdefault('MUJOCO_GL','glfw')
from pi0_repro.config import load_config
from pi0_repro.observations import aloha_observation
from pi0_repro.remote import RemotePi0
from pi0_repro.simulation import _make_env,run

out=ROOT/'outputs/motion_first_20261026';data=json.loads((out/'data_collections.json').read_text())
assert data['stage']=='full_experts_reviewed'
source=Path(data['full_expert_collection']);destination=out/'learners';destination.mkdir(exist_ok=False)
config_path=ROOT/'configs/banana_v8_control32_pi0.toml'
record={'scope':'Actual pi0 rollouts on the12new training layouts only. No validation/test observations become correction labels.',
        'complete':False,'results':[]}
ledger=destination/'comparison.json';ledger.write_text(json.dumps(record,indent=2)+'\n')
config=load_config(config_path);config.connection.connect_timeout_seconds=2
deadline=time.monotonic()+90
while True:
    try:
        with RemotePi0(config) as client:metadata=client.metadata
        break
    except (OSError,TimeoutError):
        if time.monotonic()>deadline:raise
        time.sleep(2)
    except RuntimeError as exc:
        if 'Image transport failed' not in str(exc) or 'ConnectionRefusedError' not in str(exc) or time.monotonic()>deadline:raise
        time.sleep(2)
warm=destination/'warmup';warm.mkdir();env=_make_env(config,warm)
try:
    obs,_=env.reset(seed=config.simulation.seed)
    observation=aloha_observation({'pixels':{'top':env.render()},'agent_pos':obs['agent_pos']},config.simulation.prompt,resize=False)
    prefix=np.repeat(obs['agent_pos'][None],100,axis=0);prefix[:,[6,13]]=np.clip(prefix[:,[6,13]],0,1)
    with RemotePi0(config) as client:
        for delay,blend in ((0,0),(10,20)):
            a,ms,response=client.infer({**observation,'rtc':dict(schema='pi0-rtc-v1',prefix=prefix,valid_steps=100,delay_steps=delay,blend_steps=blend,guidance_max=.5)})
            assert a.shape==(100,14) and np.isfinite(a).all()
finally:env.close()
for episode in sorted(source.glob('episode_*')):
    meta=json.loads((episode/'episode.json').read_text())
    assert meta['split']=='train' and meta['eligible_for_training'] and meta['review_status']=='accepted'
    index=len(record['results']);scene=destination/f'scene_{index:02d}.toml'
    scene.write_text('[scene]\n'+'\n'.join(f'{k} = {json.dumps(v)}' for k,v in meta['scene'].items() if v is not None)+'\n')
    config=load_config(config_path);config.simulation.display=False;config.simulation.scene_config=str(scene);config.simulation.seed=meta['reset_seed']
    print('START train learner',index+1,meta['layout_id'],flush=True)
    path=run(config);summary=json.loads((path/'summary.json').read_text())['episodes'][0]
    record['results'].append({'index':index,'source_episode':str(episode),'layout_id':meta['layout_id'],'split':'train',
                              'run_directory':str(path),'result':summary})
    ledger.write_text(json.dumps(record,indent=2)+'\n')
    print('END train learner',index+1,'success',summary['success'],flush=True)
record['complete']=True;ledger.write_text(json.dumps(record,indent=2)+'\n')
data['learner_ledger']=str(ledger);data['stage']='learners_recorded';(out/'data_collections.json').write_text(json.dumps(data,indent=2)+'\n')
print('COMPLETE train learners',len(record['results']),flush=True)

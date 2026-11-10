"""Post-hoc interface/prior prediction check; never tunes or trains ACT."""
from pathlib import Path
import json
import numpy as np
import torch
import pyarrow.parquet as pq
from act_baseline import policy,image_tensor,KEY

def main(root):
 torch.set_num_threads(4);out=root/'outputs/act_baseline_20261029';saved=torch.load(out/'train/checkpoint_020000.pt',map_location='cpu',weights_only=False);model=policy(saved['stats'],False).cuda().eval();model.load_state_dict(saved['state_dict']);data=root/'data/lerobot/20261029-135803-ik-replaced-b8ed3a/local/pi0_banana_train';mapping=json.loads((data/'source_episodes.json').read_text());rows=[];errors=[];identity=[]
 for ep in mapping['episodes']:
  t=pq.read_table(data/f"data/chunk-000/episode_{ep['lerobot_episode_index']:06d}.parquet");states=np.array(t['observation.state'].to_pylist(),np.float32);actions=np.array(t['action'].to_pylist(),np.float32);indices=[0,len(t)//3,2*len(t)//3]
  for i in indices:
   image=t[KEY][i].as_py()['bytes'];b={KEY:torch.from_numpy(image_tensor(image)).unsqueeze(0).float().cuda()/255,'observation.state':torch.from_numpy(states[i]).unsqueeze(0).cuda()};target=torch.from_numpy(actions[i:i+100].copy()).unsqueeze(0).cuda();assert target.shape==(1,100,14)
   with torch.inference_mode():
    normalized=model.normalize_inputs(b);normalized['observation.images']=[normalized[KEY]];a=model.model(normalized)[0];native=model.unnormalize_outputs({'action':a})['action'];model.reset();public=model.select_action(b);errors.append(float((public-native[:,0]).abs().max()));nt=model.normalize_targets({'action':target})['action'];rt=model.unnormalize_outputs({'action':nt})['action'];identity.append(float((rt-target).abs().max()));rows.append(dict(episode=ep['lerobot_episode_index'],frame=i,prior_normalized_l1=float((a-nt).abs().mean()),first_action_arm_mae_rad=float((native[0,0,[0,1,2,3,4,5,7,8,9,10,11,12]]-target[0,0,[0,1,2,3,4,5,7,8,9,10,11,12]]).abs().mean()),first_action_gripper_mae=float((native[0,0,[6,13]]-target[0,0,[6,13]]).abs().mean())))
 result=dict(complete=True,samples=len(rows),public_select_action_vs_server_formula_max_error=max(errors),normalization_roundtrip_max_error=max(identity),mean_prior_normalized_l1=float(np.mean([r['prior_normalized_l1'] for r in rows])),rows=rows,passed_interface=max(errors)<1e-6 and max(identity)<1e-6,scope='Post-hoc66 fixed train-only teacher-state samples, not held-out validation or a new policy attempt. Prior-zero-latent chunk prediction vs actual labels; interface equality/normalization does not establish global semantic correctness or closed-loop reliability.')
 p=out/'prediction_check.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
if __name__=='__main__':import sys;main(Path(sys.argv[1]))

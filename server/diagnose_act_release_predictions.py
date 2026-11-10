"""Fixed-checkpoint teacher/learner release queries, without optimization."""
from pathlib import Path
import json,hashlib
import torch,numpy as np
from act_baseline import policy,image_tensor,KEY

def main(root):
 torch.set_num_threads(4);out=root/'outputs/act_release_diagnosis_20261029';checkpoint=root/'outputs/act_baseline_20261029/train/checkpoint_020000.pt';saved=torch.load(checkpoint,map_location='cpu',weights_only=False);model=policy(saved['stats'],False).cuda().eval();model.load_state_dict(saved['state_dict']);queries=json.loads((out/'queries.json').read_text());rows=[]
 for q in queries:
  pixels=(out/q['image']).read_bytes();b={KEY:torch.from_numpy(image_tensor(pixels)).unsqueeze(0).float().cuda()/255,'observation.state':torch.tensor(q['state'],dtype=torch.float32).unsqueeze(0).cuda()}
  with torch.inference_mode():
   x=model.normalize_inputs(b);x['observation.images']=[x[KEY]];p=model.model(x)[0];a=model.unnormalize_outputs({'action':p})['action'][0].cpu().numpy()
  g=a[:,13];row={k:q[k] for k in ['kind','case','step']};row.update(gripper_chunk=g.tolist(),first25_gripper_range=[float(g[:25].min()),float(g[:25].max())],future75_gripper_range=[float(g[25:].min()),float(g[25:].max())],first_open_offset=next((i for i,v in enumerate(g) if v>=.8),None),image_sha256=hashlib.sha256(pixels).hexdigest())
  if q['kind']=='teacher':
   target=np.array(q['target_actions']);row.update(phase=q['phase'],relative_release_step=q['step']-q['release_start'],target_first25_range=[float(target[:25,13].min()),float(target[:25,13].max())],target_first_open_offset=next((i for i,v in enumerate(target[:,13]) if v>=.8),None),gripper_first25_mae=float(np.mean(abs(g[:25]-target[:25,13]))),gripper_100_mae=float(np.mean(abs(g-target[:,13]))))
  else:row['recorded_first25_range']=[min(q['executed_gripper_targets']),max(q['executed_gripper_targets'])];row['recomputed_first25_max_error']=float(np.max(abs(g[:25]-q['executed_gripper_targets'])))
  rows.append(row)
 result=dict(complete=True,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),queries_sha256=hashlib.sha256((out/'queries.json').read_bytes()).hexdigest(),rows=rows,scope='Post-hoc fixed-weight inference only. Teacher states are from accepted train data; learner states from exact recorded-command replay. No gradients, no runtime deployment, no new policy episode. Raw future predictions beyond25 are not executed by baseline.')
 p=out/'predictions.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n');print('COMPLETE',len(rows))
if __name__=='__main__':import sys;main(Path(sys.argv[1]))

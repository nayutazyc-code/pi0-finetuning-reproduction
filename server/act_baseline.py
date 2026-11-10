"""Project adapter around the pinned, unmodified LeRobot ACT implementation."""
import argparse,hashlib,io,json,os,time,random
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
import torch
from torch.utils.data import Dataset,DataLoader

KEY='observation.images.top'
ROI=(275,110,540,345)

def image_tensor(image):
 if isinstance(image,bytes):image=Image.open(io.BytesIO(image)).convert('RGB')
 elif isinstance(image,np.ndarray):image=Image.fromarray(image)
 assert image.size==(640,480)
 image=ImageOps.pad(image.crop(ROI),(224,224),method=Image.Resampling.BILINEAR,color=0,centering=(.5,.5))
 return np.asarray(image,dtype=np.uint8).transpose(2,0,1).copy()

class Episodes(Dataset):
 def __init__(self,root,cache):
  import pyarrow.parquet as pq
  mapping=json.loads((root/'source_episodes.json').read_text());assert mapping['source_split']=='train' and all(r['split']=='train' for r in mapping['episodes'])
  self.ends=[];states=[];actions=[];n=sum(r['frames'] for r in mapping['episodes']);cache.mkdir(exist_ok=True,parents=True)
  path=cache/'rgb.npy';assert not path.exists(),'Preserve prior cache/run'
  images=np.lib.format.open_memmap(path,mode='w+',dtype=np.uint8,shape=(n,3,224,224));pos=0;checks=[]
  for row in mapping['episodes']:
   p=root/f"data/chunk-000/episode_{row['lerobot_episode_index']:06d}.parquet";t=pq.read_table(p);s=np.array(t['observation.state'].to_pylist(),dtype=np.float32);a=np.array(t['action'].to_pylist(),dtype=np.float32)
   assert s.shape==a.shape==(row['frames'],14) and np.isfinite(s).all() and np.isfinite(a).all();assert np.allclose(np.array(t['timestamp'].to_pylist()),np.arange(len(s))*.02,atol=2e-6)
   for j,v in enumerate(t[KEY].to_pylist()):images[pos+j]=image_tensor(v['bytes'])
   states.append(s);actions.append(a);pos+=len(s);self.ends.extend([pos]*len(s));checks.append(dict(episode=row['lerobot_episode_index'],sha256=hashlib.sha256(p.read_bytes()).hexdigest(),frames=len(s)))
  images.flush();self.images=np.load(path,mmap_mode='r');self.state=torch.from_numpy(np.concatenate(states));self.action=torch.from_numpy(np.concatenate(actions));self.ends=np.array(self.ends)
  stats={k:dict(mean=v.mean(0).numpy(),std=v.std(0,unbiased=False).clamp_min(1e-4).numpy()) for k,v in [('observation.state',self.state),('action',self.action)]};stats[KEY]=dict(mean=np.array([.485,.456,.406],np.float32).reshape(3,1,1),std=np.array([.229,.224,.225],np.float32).reshape(3,1,1));self.stats=stats
  (cache/'audit.json').write_text(json.dumps(dict(frames=pos,episodes=len(checks),sources=checks,alignment='Current image/state with same indexed executed absolute action. Future indices clipped within episode; padding masked.',preprocessing='Fixed original-pi0 ROI, PIL bilinear fit+black letterbox224; same function at serving. ImageNet visual stats; train-only state/action stats.'),indent=2))
  for i in [0,*np.flatnonzero(np.diff(self.ends))]:
   b=self[int(i)];assert b['action'].shape==(100,14) and torch.equal(b['action'][0],self.action[i]);assert b['action_is_pad'].dtype==torch.bool
 def __len__(self):return len(self.state)
 def __getitem__(self,i):
  js=np.arange(i,i+100);pad=js>=self.ends[i];js=np.minimum(js,self.ends[i]-1)
  return {KEY:torch.from_numpy(self.images[i].copy()).float()/255,'observation.state':self.state[i],'action':self.action[js],'action_is_pad':torch.from_numpy(pad)}

def policy(stats,pretrained=True):
 from lerobot.common.policies.act.configuration_act import ACTConfig
 from lerobot.common.policies.act.modeling_act import ACTPolicy
 from lerobot.configs.types import PolicyFeature,FeatureType
 cfg=ACTConfig(input_features={KEY:PolicyFeature(FeatureType.VISUAL,(3,224,224)),'observation.state':PolicyFeature(FeatureType.STATE,(14,))},output_features={'action':PolicyFeature(FeatureType.ACTION,(14,))},chunk_size=100,n_action_steps=25,pretrained_backbone_weights='ResNet18_Weights.IMAGENET1K_V1' if pretrained else None,device='cuda')
 return ACTPolicy(cfg,dataset_stats=stats)

def train(root,out,updates,probe):
 torch.set_num_threads(4);torch.manual_seed(54);np.random.seed(54);random.seed(54);torch.backends.cudnn.benchmark=False
 out.mkdir(parents=True,exist_ok=False);data=Episodes(root/'data/lerobot/20261029-135803-ik-replaced-b8ed3a/local/pi0_banana_train',out/'cache');model=policy(data.stats).cuda();opt=torch.optim.AdamW(model.get_optim_params(),lr=1e-5,weight_decay=1e-4)
 torch.cuda.reset_peak_memory_stats();loader=DataLoader(data,batch_size=8,shuffle=True,num_workers=4,pin_memory=True,drop_last=True,persistent_workers=True);iterator=iter(loader);log=(out/'metrics.jsonl').open('x');start=time.monotonic();times=[]
 try:
  for step in range(updates):
   tic=time.monotonic()
   try:batch=next(iterator)
   except StopIteration:iterator=iter(loader);batch=next(iterator)
   batch={k:v.cuda(non_blocking=True) for k,v in batch.items()};model.train();opt.zero_grad(set_to_none=True);loss,metrics=model(batch);assert torch.isfinite(loss);loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step();torch.cuda.synchronize();times.append(time.monotonic()-tic)
   if step%100==0 or step==updates-1:
    record=dict(step=step+1,loss=float(loss),grad_norm=float(gn),**metrics,elapsed_seconds=time.monotonic()-start,update_seconds=float(np.median(times[-100:])),cuda_peak_mib=torch.cuda.max_memory_allocated()/1024**2);log.write(json.dumps(record)+'\n');log.flush();print(json.dumps(record),flush=True)
   if not probe and ((step+1)%5000==0 or step==updates-1):
    destination=out/f'checkpoint_{step+1:06d}.pt';tmp=destination.with_suffix('.tmp');torch.save(dict(state_dict=model.state_dict(),stats=data.stats,step=step+1,lerobot_commit='0cf864870cf29f4738d3ade893e6fd13fbd7cdb5'),tmp);tmp.rename(destination)
 finally:log.close()
 (out/'completion.json').write_text(json.dumps(dict(complete=True,updates=updates,probe=probe,seconds=time.monotonic()-start,peak_mib=torch.cuda.max_memory_allocated()/1024**2,median_last_updates_seconds=float(np.median(times[-100:]))),indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--updates',type=int,default=20000);p.add_argument('--probe',action='store_true');a=p.parse_args();train(a.root,a.out,a.updates,a.probe)

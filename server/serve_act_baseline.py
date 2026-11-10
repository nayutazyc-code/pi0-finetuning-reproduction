"""Separate ACT endpoint; never advertises pi0/RTC compatibility."""
import argparse,asyncio,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from openpi_client import msgpack_numpy
from websockets.asyncio.server import serve
from act_baseline import policy,image_tensor,KEY
from image_transport_proxy import decode_images

async def main(checkpoint,port):
 torch.set_num_threads(4);saved=torch.load(checkpoint,map_location='cpu',weights_only=False);model=policy(saved['stats'],pretrained=False).cuda().eval();model.load_state_dict(saved['state_dict']);digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
 meta=dict(model_family='act',action_dim=14,action_horizon=100,checkpoint=str(checkpoint),checkpoint_sha256=digest,lerobot_commit=saved['lerobot_commit'],step=saved['step'],action_semantics='native_aloha_absolute',image_transport=['png'],crop=[275,110,540,345],preprocessing='PIL bilinear letterbox224',temporal_ensemble=False)
 async def handler(ws):
  pack=msgpack_numpy.Packer();await ws.send(pack.pack(meta))
  async for raw in ws:
   obs,transport=decode_images(msgpack_numpy.unpackb(raw));state=np.asarray(obs['state'],np.float32);assert state.shape==(14,)
   image=obs['images']['cam_high'];assert image.shape==(3,480,640)
   b={KEY:torch.from_numpy(image_tensor(image.transpose(1,2,0))).unsqueeze(0).float().cuda()/255,'observation.state':torch.from_numpy(state).unsqueeze(0).cuda()};start=time.perf_counter()
   with torch.inference_mode():
    b=model.normalize_inputs(b);b['observation.images']=[b[KEY]];a=model.model(b)[0];a=model.unnormalize_outputs({'action':a})['action'][0].cpu().numpy()
   assert a.shape==(100,14) and np.isfinite(a).all();await ws.send(pack.pack(dict(actions=a,server_timing={'infer_ms':(time.perf_counter()-start)*1000},transport=transport)))
 async with serve(handler,'127.0.0.1',port,compression=None,max_size=16*1024*1024):
  print('ACT_READY',json.dumps(meta),flush=True);await asyncio.Future()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--port',type=int,default=8002);a=p.parse_args();asyncio.run(main(a.checkpoint,a.port))

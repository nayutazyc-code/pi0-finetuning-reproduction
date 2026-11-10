import sys
from pathlib import Path
import io,unittest
import numpy as np
import torch
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'server'))
from act_baseline import Episodes,image_tensor

class ACTDataTest(unittest.TestCase):
 def test_actions_start_at_observation_and_never_cross_episode(self):
  d=Episodes.__new__(Episodes);d.state=torch.zeros(5,14);d.action=torch.arange(5).float()[:,None].expand(5,14);d.ends=np.array([3,3,3,5,5]);d.images=np.zeros((5,3,224,224),dtype=np.uint8)
  b=d[1];self.assertEqual(b['action'][:,0].tolist(),[1.]+[2.]*99);self.assertEqual(b['action_is_pad'][:3].tolist(),[False,False,True]);self.assertEqual(d[3]['action'][:3,0].tolist(),[3.,4.,4.]);self.assertEqual(d[4]['action_is_pad'].sum(),99)
 def test_png_and_wire_pixels_have_identical_preprocessing(self):
  a=np.zeros((480,640,3),np.uint8);a[110:345,275:540]=[120,70,210];buffer=io.BytesIO();Image.fromarray(a).save(buffer,format='PNG');from_png=image_tensor(buffer.getvalue());from_wire=image_tensor(a);self.assertTrue(np.array_equal(from_png,from_wire));self.assertEqual(from_png.shape,(3,224,224));self.assertTrue(np.array_equal(from_png[:,112,112],[120,70,210]));self.assertTrue(np.all(from_png[:,0,:]==0))
if __name__=='__main__':unittest.main()

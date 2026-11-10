"""Single-camera ROI adaptation for original pi0; masks and action semantics stay unchanged."""
import dataclasses
from openpi import transforms
from openpi.training.config import LeRobotAlohaDataConfig
from banana_model import BananaPi0, BananaPi0Config
import flax.nnx as nnx


@dataclasses.dataclass(frozen=True)
class CropBaseImage:
    bounds: tuple[int,int,int,int]
    def __call__(self,data):
        image=data["image"]["base_0_rgb"]
        if image.shape[:2]!=(480,640):
            raise ValueError(f"ROI policy requires raw 640x480 camera input, got {image.shape}")
        x0,y0,x1,y1=self.bounds
        data["image"]["base_0_rgb"]=image[y0:y1,x0:x1].copy()
        # Missing wrist views stay black and masked; use matching sizes for normal preprocessing.
        for name in ("left_wrist_0_rgb","right_wrist_0_rgb"):
            if not data["image_mask"][name]:
                data["image"][name]=data["image"][name][y0:y1,x0:x1].copy()
        return data


@dataclasses.dataclass(frozen=True)
class ROIAlohaDataConfig(LeRobotAlohaDataConfig):
    crop_bounds: tuple[int,int,int,int]=(275,110,540,345)
    def create(self,assets_dirs,model_config):
        config=super().create(assets_dirs,model_config)
        return dataclasses.replace(config,data_transforms=config.data_transforms.push(inputs=[CropBaseImage(self.crop_bounds)]))


@dataclasses.dataclass(frozen=True)
class ROIPi0Config(BananaPi0Config):
    def create(self,rng):
        return ROIPi0(self,rngs=nnx.Rngs(rng))


class ROIPi0(BananaPi0):
    def embed_prefix(self,obs):
        # Wrist cameras are permanently absent in this dataset. Removing masked slots
        # preserves valid-token positions and attention, and avoids their unused encoders.
        obs=obs.replace(images={"base_0_rgb":obs.images["base_0_rgb"]},
                        image_masks={"base_0_rgb":obs.image_masks["base_0_rgb"]})
        return super().embed_prefix(obs)

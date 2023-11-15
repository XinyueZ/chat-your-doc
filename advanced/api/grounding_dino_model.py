import os

from groundingdino.util.inference import Model
from rich import print


class GroundingDINOModel:
    box_threshold: float = 0.35
    text_threshold: float = 0.25

    def __init__(self, cfg_url, download_url, name, device):
        self.cfg_url = cfg_url
        self.download_url = download_url
        self.name = name
        self.device = device

    def setup(self, box_threshold: float = 0.35, text_threshold: float = 0.25):
        os.system(f"wget -nc  -P cfgs {self.cfg_url}")
        os.system(f"wget -nc  -P models  {self.download_url}")

        self.gddino = Model(
            f"./cfgs/{self.cfg_url.split('/')[-1]}",
            f"./models/{self.download_url.split('/')[-1]}",
            device=self.device,
        )

        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        print(
            f"GroundingDINO model loaded, type: {self.name}, device: {self.device}, box_threshold: {self.box_threshold}, text_threshold: {self.text_threshold}"
        )

        return self

    def __call__(self, for_classes, **kwargs):
        kwargs["box_threshold"] = self.box_threshold
        kwargs["text_threshold"] = self.text_threshold
        if for_classes:
            return self.gddino.predict_with_classes(**kwargs)
        else:
            return self.gddino.predict_with_caption(**kwargs)

    @staticmethod
    def create_instance(device, groundingDINO_type):
        groundingDINO_models = {
            "swint_ogc": GroundingDINOModel(
                "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
                "swint_ogc",
                device=device,
            ),
            "swinb_cogcoor": GroundingDINOModel(
                "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinB_cfg.py",
                "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth",
                "swinb_cogcoor",
                device=device,
            ),
        }

        return groundingDINO_models.get(groundingDINO_type, None)

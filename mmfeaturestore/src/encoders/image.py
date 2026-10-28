"""Image features: ResNet-18 pretrained on ImageNet, without its classification layer (512 numbers,
L2-normalized). The weights (45 MB) download on first use. GTZAN's images are mel spectrograms."""
import logging

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


class ImageEncoder:
    def __init__(self, pretrained: bool = True):
        self.logger = logging.getLogger(__name__)
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        self.model = torch.nn.Sequential(*(list(backbone.children())[:-1])).eval()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def encode(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        with torch.no_grad():
            features = self.model(self.transform(image).unsqueeze(0)).flatten().numpy()
        norm = np.linalg.norm(features)
        return features / norm if norm > 0 else features

import logging
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

class ImageEncoder:
    """Encodes images into feature vectors"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Load pre-trained model (ResNet18)
        self.model = models.resnet18(weights=None)
        
        # Remove the final fully connected layer
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Define preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def encode(self, image_path: str) -> np.ndarray:
        """Extract features from image file"""
        self.logger.info(f"Processing image file: {image_path}")
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)
        
        # Get features
        with torch.no_grad():
            features = self.model(image_tensor)
            features = features.squeeze().numpy()
        
        # Normalize features
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
            
        self.logger.info(f"Generated image features with shape: {features.shape}")
        return features 
import torch
import torchvision.transforms as T
import torchreid
import cv2
import numpy as np
from utils.logger import get_system_logger

class ReIDEngine:
    def __init__(self, device: int):
        get_system_logger().info("[ReID] Loading OSNet pretrained...")
        self.device = device
        self.model = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=1000,
            loss='softmax',
            pretrained=True
        ).to(device).eval()
        get_system_logger().info("[ReID] Loaded.")

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess(self, crop: np.ndarray):
        """Prepare a BGR numpy array crop into a tensor for ReID"""
        return self.transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    def extract_features(self, tensors: list) -> np.ndarray:
        """Extract features back to CPU numpy array"""
        if not tensors:
            return np.array([])
        
        # Mini-batching chống đầy tràn VRAM (CUDACachingAllocator error 12)
        mini_batch = 8
        all_features = []
        for i in range(0, len(tensors), mini_batch):
            chunk = tensors[i:i+mini_batch]
            batch_t = torch.stack(chunk).to(self.device)
            features = self.model(batch_t).cpu().numpy()
            all_features.append(features)
            
        batch_features = np.concatenate(all_features, axis=0)
        # Normalize
        batch_features = batch_features / (np.linalg.norm(batch_features, axis=1, keepdims=True) + 1e-6)
        return batch_features

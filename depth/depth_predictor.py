import torch
import cv2
import numpy as np

class DepthPredictor:
    """
    Predicts monocular depth for each RGB frame.
    """

    def __init__(self, device="cuda"):
        self.device = device
        self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Large").to(device)
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
        self.model.eval()

    def predict(self, rgb):
        """
        Returns raw depth (not metric).
        """
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        batch = self.transform(img).to(self.device)

        with torch.no_grad():
            depth = self.model(batch).squeeze().cpu().numpy()

        depth = depth - depth.min()
        depth = depth / depth.max()
        depth = depth * 5.0  # scale to ~5 meters
        return depth

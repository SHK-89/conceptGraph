import torch
import open_clip
from PIL import Image
import numpy as np


class CLIPEncoder:
    """
    Extracts semantic features for each object crop using CLIP.
    """

    def __init__(self, model_name="ViT-B-32", device=None):


        self.device = device if device is not None else get_best_device()

        print(f"Initializing CLIP on device: {self.device}")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained="laion2b_s34b_b79k"
        )
        self.model.to(self.device)
        self.model.eval()

    def encode_masked(self, rgb, mask):
        """
        Returns the CLIP embedding of the object defined by mask.
        """
        rgb_masked = (rgb * mask[..., None]).astype(np.uint8)
        img = Image.fromarray(rgb_masked)
        img = self.preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.model.encode_image(img)

        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze().cpu().numpy()

def get_best_device():
    # Apple M-series GPU
    if torch.backends.mps.is_available():
        return "mps"

    # NVIDIA GPU
    if torch.cuda.is_available():
        return "cuda"

    # Default fallback
    return "cpu"

import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as T

from salad.eval import load_model

class ImageRetrieval:
    def __init__(self, input_size=224, device=None):
        """
        Initializes the DINOv2-SALAD model for Global Place Recognition.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SALAD] Loading Loop Closure Model on {self.device}...")
        
        # 1. Load Model (Automatically finds your downloaded weights)
        self.model = load_model()
        self.model.to(self.device)
        self.model.eval()
        
        # 2. Standard DINOv2 / ImageNet Normalization Stats
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]
        
        # 3. Vision Transformer Pipeline
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        
        print("[SALAD] Model loaded and ready.")

    @torch.no_grad()
    def get_single_embeding(self, img_np: np.ndarray) -> torch.Tensor:
        """
        Takes an RGB numpy array (H, W, 3), runs it through the Vision Transformer, 
        and returns a L2-normalized 1D embedding tensor for cosine similarity matching.
        """
        img_tensor = self.transform(img_np).unsqueeze(0).to(self.device)
        embedding = self.model(img_tensor)
        
        if embedding.dim() > 2:
            embedding = embedding.view(1, -1)
            
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding
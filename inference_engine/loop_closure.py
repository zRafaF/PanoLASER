import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as T

class ImageRetrieval:
    def __init__(self, input_size=224, device=None):
        """
        Initializes the DINOv2-SALAD model for Global Place Recognition.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SALAD] Loading Loop Closure Model on {self.device}...")
        
        # --- THE FIX: Restored the Torch Hub load that you successfully got working ---
        print("[SALAD] Fetching architecture and weights from Torch Hub...")
        self.model = torch.hub.load('serizba/salad', 'dinov2_salad', trust_repo=True)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Standard DINOv2 / ImageNet Normalization Stats
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]
        
        # Vision Transformer Pipeline
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
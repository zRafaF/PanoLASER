import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as T

# Load the SALAD model from your local environment setup
try:
    from salad.eval import load_model
except ImportError:
    print("WARNING: Could not import salad.eval. Please ensure SALAD is installed in your venv.")

class ImageRetrieval:
    def __init__(self, input_size=224, device=None):
        """
        Initializes the DINOv2-SALAD model for Global Place Recognition.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SALAD] Loading Loop Closure Model on {self.device}...")
        
        # 1. Load Model
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
        and returns a normalized 1D embedding tensor for cosine similarity matching.
        """
        # Transform image to PyTorch tensor
        img_tensor = self.transform(img_np).unsqueeze(0).to(self.device)
        
        # Extract deep feature embedding
        embedding = self.model(img_tensor)
        
        # Flatten to 1D if necessary
        if embedding.dim() > 2:
            embedding = embedding.view(1, -1)
            
        # L2 Normalize the embedding so Cosine Similarity behaves like a strict percentage
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding

    @torch.no_grad()
    def get_batch_embeddings(self, imgs_np: list) -> torch.Tensor:
        """
        Optional helper if you ever want to embed multiple frames at once.
        """
        tensor_list = [self.transform(img) for img in imgs_np]
        batch_tensor = torch.stack(tensor_list).to(self.device)
        
        embeddings = self.model(batch_tensor)
        
        if embeddings.dim() > 2:
            embeddings = embeddings.view(embeddings.size(0), -1)
            
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
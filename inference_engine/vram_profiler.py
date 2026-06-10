import torch

class VRAMProfiler:
    def __init__(self, device=0):
        self.device = device
        
    def start(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        
    def stop(self):
        if not torch.cuda.is_available():
            return 0, 0, 0, 0
            
        # PyTorch-only tracked allocations
        pt_alloc = torch.cuda.max_memory_allocated(self.device) / (1024**3)
        pt_res = torch.cuda.max_memory_reserved(self.device) / (1024**3)
        
        # True System-level VRAM limits (Catches Nvblox C++ allocations)
        free, total = torch.cuda.mem_get_info(self.device)
        sys_used = (total - free) / (1024**3)
        sys_total = total / (1024**3)
        
        return pt_alloc, pt_res, sys_used, sys_total
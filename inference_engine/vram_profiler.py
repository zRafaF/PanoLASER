import torch
import threading
import time

class VRAMProfiler:
    def __init__(self, device=0):
        self.device = device
        self.running = False
        self.thread = None
        self.history_alloc = []
        self.history_res = []
        
    def start(self):
        torch.cuda.reset_peak_memory_stats(self.device)
        self.running = True
        self.history_alloc.clear()
        self.history_res.clear()
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()
        
    def _poll(self):
        while self.running:
            alloc = torch.cuda.memory_allocated(self.device) / (1024**3)
            res = torch.cuda.memory_reserved(self.device) / (1024**3)
            self.history_alloc.append(alloc)
            self.history_res.append(res)
            time.sleep(0.05) # Poll at 20Hz
            
    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()
        
        if not self.history_alloc:
            return 0, 0, 0, 0
            
        avg_alloc = sum(self.history_alloc) / len(self.history_alloc)
        max_alloc = torch.cuda.max_memory_allocated(self.device) / (1024**3)
        avg_res = sum(self.history_res) / len(self.history_res)
        max_res = torch.cuda.max_memory_reserved(self.device) / (1024**3)
        
        return avg_alloc, max_alloc, avg_res, max_res
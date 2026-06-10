"""OPT-28: CUDA data prefetcher"""
import torch

class CUDAPrefetcher:
    def __init__(self, loader, device="cuda"):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_data = None
            return
        with torch.cuda.stream(self.stream):
            if isinstance(self.next_data, dict):
                self.next_data = {k: v.to(self.device, non_blocking=True) for k, v in self.next_data.items() if isinstance(v, torch.Tensor)}

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        if data is None:
            raise StopIteration
        self.preload()
        return data

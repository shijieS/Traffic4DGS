"""OPT-33: GPU profiler"""
import torch, time
from contextlib import contextmanager

class GPUProfiler:
    def __init__(self):
        self.records = {}
        self.active = False

    @contextmanager
    def profile(self, name):
        if not self.active:
            yield; return
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        m0 = torch.cuda.memory_allocated()/1024**2
        yield
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        dm = torch.cuda.memory_allocated()/1024**2 - m0
        self.records.setdefault(name, {"t":[], "m":[]})["t"].append(dt)
        self.records[name]["m"].append(dm)

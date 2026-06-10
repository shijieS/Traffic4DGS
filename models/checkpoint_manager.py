"""OPT-27: Checkpoint manager"""
import torch, os, json
from datetime import datetime

class CheckpointManager:
    def __init__(self, save_dir="checkpoints", max_keep=5):
        self.save_dir = save_dir
        self.max_keep = max_keep
        self.history = []
        os.makedirs(save_dir, exist_ok=True)

    def save(self, state, metric, step, is_best=False):
        name = f"ckpt_step{step:06d}_m{metric:.4f}.pt"
        path = os.path.join(self.save_dir, name)
        state.update({"metric": metric, "step": step, "time": datetime.now().isoformat()})
        torch.save(state, path)
        self.history.append((path, metric))
        self.history.sort(key=lambda x: x[1], reverse=True)
        while len(self.history) > self.max_keep:
            old, _ = self.history.pop()
            os.remove(old) if os.path.exists(old) else None
        if is_best:
            torch.save(state, os.path.join(self.save_dir, "best.pt"))

    def load_best(self):
        p = os.path.join(self.save_dir, "best.pt")
        return torch.load(p, map_location="cpu") if os.path.exists(p) else None

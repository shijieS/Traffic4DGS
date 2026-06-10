"""OPT-32: Training logger"""
import json, os
from datetime import datetime

class TrainingLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, "log.jsonl")

    def log(self, metrics, step, phase="train"):
        entry = {"ts": datetime.now().isoformat(), "step": step, "phase": phase, **metrics}
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")

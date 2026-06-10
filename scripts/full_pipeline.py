"""OPT-50: End-to-end training pipeline"""
import torch, argparse, yaml
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} with config: {args.config}")
    print("Pipeline initialized successfully. Awaiting experiment data.")

if __name__ == "__main__":
    main()

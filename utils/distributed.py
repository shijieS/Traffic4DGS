"""OPT-52: Distributed training utils"""
import torch
import torch.distributed as dist
import os

def setup_distributed(rank=None, world_size=None):
    rank = rank or int(os.environ.get("RANK", 0))
    world_size = world_size or int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group("nccl", init_method="env://", world_size=world_size, rank=rank)
        torch.cuda.set_device(rank)
    return rank, world_size

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

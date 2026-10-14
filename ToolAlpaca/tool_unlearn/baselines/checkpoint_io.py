"""Bound host-memory use of DeepSpeed 0.12.6 checkpoint IO on shared storage."""
import ctypes
import gc
import os
from pathlib import Path

import torch


def release_cpu_memory():
    gc.collect()
    ctypes.CDLL(None).malloc_trim(0)


def durable_save(state, path):
    """Flush each completed file and release its reclaimable page cache."""
    path = Path(path)
    link = None
    optimizer_root = os.environ.get("RETRAIN_OPTIMIZER_ROOT")
    if optimizer_root and path.name.endswith("optim_states.pt"):
        link = path
        path = Path(optimizer_root) / path.parent.parent.name / path.parent.name / path.name
        path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as handle:
        torch.save(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
        os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    temporary.replace(path)
    if link is not None:
        temporary_link = link.with_name(link.name + ".partial-link")
        temporary_link.symlink_to(path)
        temporary_link.replace(link)
    release_cpu_memory()


def install():
    from deepspeed.runtime.engine import DeepSpeedEngine
    from deepspeed.runtime.checkpoint_engine.torch_checkpoint_engine import TorchCheckpointEngine

    if getattr(DeepSpeedEngine, "_retain_serial_io", False):
        return
    original = DeepSpeedEngine._save_zero_checkpoint

    def serial_optimizer_save(self, save_path, tag):
        # This method contains only per-rank serialization, no collectives.
        # Keep the outer DeepSpeed save_checkpoint collective on every rank.
        release_cpu_memory()
        for rank in range(torch.distributed.get_world_size()):
            torch.distributed.barrier()
            if torch.distributed.get_rank() == rank:
                print(f"[checkpoint-io] saving optimizer rank={rank} tag={tag}", flush=True)
                original(self, save_path, tag)
                release_cpu_memory()
        torch.distributed.barrier()

    def save(self, state_dict, path):
        durable_save(state_dict, path)

    DeepSpeedEngine._save_zero_checkpoint = serial_optimizer_save
    TorchCheckpointEngine.save = save
    DeepSpeedEngine._retain_serial_io = True

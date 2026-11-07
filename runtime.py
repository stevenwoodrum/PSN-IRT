"""Portable device selection shared by training notebooks and audit tools."""

import torch


def select_device(requested="auto"):
    if requested == "auto":
        requested = ("cuda" if torch.cuda.is_available() else
                     "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(requested)
    if device.type not in ("cpu", "cuda", "mps"):
        raise ValueError("Choose auto, cpu, cuda, or mps.")
    # Fail clearly for an explicitly requested unavailable accelerator.
    probe = torch.ones(1, device=device) + 1
    assert probe.item() == 2
    return str(device)


def seed_device(seed, device):
    kind = torch.device(device).type
    if kind == "mps":
        torch.mps.manual_seed(seed)
    elif kind == "cuda":
        torch.cuda.manual_seed_all(seed)


def allocated_bytes(device):
    kind = torch.device(device).type
    if kind == "mps":
        return torch.mps.current_allocated_memory()
    if kind == "cuda":
        return torch.cuda.memory_allocated(torch.device(device))
    return None


def clear_cache(device):
    kind = torch.device(device).type
    if kind == "mps":
        torch.mps.empty_cache()
    elif kind == "cuda":
        torch.cuda.empty_cache()

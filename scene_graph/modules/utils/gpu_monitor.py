import torch

def get_gpu_mem():
    """
    Returns (used_GB, total_GB) for the current GPU.
    Works only when CUDA is available.
    """
    if not torch.cuda.is_available():
        return (0.0, 0.0)

    device = torch.cuda.current_device()
    used = torch.cuda.memory_allocated(device) / (1024 ** 3)
    total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    return used, total


def format_time(seconds):
    """Convert seconds to H:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

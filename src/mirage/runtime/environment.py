"""Record the environment actually used by a worker."""

import os
import hashlib
from pathlib import Path
import platform
import subprocess
import sys


def describe(device: str) -> dict:
    import torch

    package = Path(__file__).resolve().parents[1]
    code_hash = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        code_hash.update(str(path.relative_to(package)).encode())
        code_hash.update(path.read_bytes())
    result = {
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
        "device": device, "gpu": None, "driver": None,
        "mirage_code_sha256": code_hash.hexdigest(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable in this Python environment")
        properties = torch.cuda.get_device_properties(0)
        result["gpu"] = {"name": properties.name, "memory": properties.total_memory,
                         "capability": list(torch.cuda.get_device_capability(0))}
        try:
            result["driver"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                timeout=10, text=True, stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return result


def configure() -> None:
    import torch

    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")
    # Unsupported deterministic operators remain recorded exceptions, not bugs.
    torch.use_deterministic_algorithms(True)

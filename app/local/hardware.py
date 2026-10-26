"""System hardware inspection: RAM, GPU VRAM, and model fit estimation."""

from __future__ import annotations
import shutil
import subprocess
from typing import Any, Dict, List, Optional
import psutil


def get_hardware_info() -> Dict[str, Any]:
    """Inspect CPU, System RAM, and NVIDIA GPU VRAM."""
    mem = psutil.virtual_memory()
    total_ram_gb = round(mem.total / (1024 ** 3), 1)
    avail_ram_gb = round(mem.available / (1024 ** 3), 1)

    cpu_cores = psutil.cpu_count(logical=False) or psutil.cpu_count() or 4
    cpu_threads = psutil.cpu_count(logical=True) or cpu_cores

    gpus: List[Dict[str, Any]] = []
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            res = subprocess.run(
                [nvidia_smi, "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=3,
                check=True,
            )
            for line in res.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    gpus.append({
                        "name": parts[0],
                        "total_vram_mb": int(parts[1]),
                        "free_vram_mb": int(parts[2]),
                        "total_vram_gb": round(int(parts[1]) / 1024, 1),
                        "free_vram_gb": round(int(parts[2]) / 1024, 1),
                    })
        except Exception:
            pass

    return {
        "ram": {
            "total_gb": total_ram_gb,
            "available_gb": avail_ram_gb,
        },
        "cpu": {
            "cores": cpu_cores,
            "threads": cpu_threads,
        },
        "gpus": gpus,
        "has_gpu": len(gpus) > 0,
    }


def estimate_model_fit(size_bytes: int, hw_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Estimate whether a model will fit in VRAM or RAM and suggest parameters."""
    if hw_info is None:
        hw_info = get_hardware_info()

    size_gb = size_bytes / (1024 ** 3)
    gpus = hw_info.get("gpus", [])
    total_vram_gb = sum(g.get("free_vram_gb", 0) for g in gpus)
    avail_ram_gb = hw_info.get("ram", {}).get("available_gb", 0)

    # Estimate KV cache overhead for 8k context (approx 1.5 - 2.5 GB depending on model)
    kv_overhead_gb = 1.8
    total_needed_gb = size_gb + kv_overhead_gb

    if total_vram_gb >= total_needed_gb:
        fit = "Full GPU"
        status = "excellent"
        suggested_ngl = 99
        suggested_ctx = 16384 if total_vram_gb >= total_needed_gb + 2.0 else 8192
    elif (total_vram_gb + avail_ram_gb) >= total_needed_gb:
        if total_vram_gb > 2.0:
            fit = "Hybrid (GPU + RAM)"
            status = "good"
            # Proportion of layers on GPU
            ratio = max(0.1, min(0.9, (total_vram_gb - 0.5) / size_gb))
            suggested_ngl = int(32 * ratio)
            suggested_ctx = 8192
        else:
            fit = "CPU / System RAM"
            status = "moderate"
            suggested_ngl = 0
            suggested_ctx = 8192
    else:
        fit = "Insufficient RAM/VRAM"
        status = "warning"
        suggested_ngl = 0
        suggested_ctx = 4096

    return {
        "model_size_gb": round(size_gb, 2),
        "total_needed_gb": round(total_needed_gb, 2),
        "fit": fit,
        "status": status,
        "suggested_ngl": suggested_ngl,
        "suggested_ctx": suggested_ctx,
    }

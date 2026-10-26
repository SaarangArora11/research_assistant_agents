"""Scanner for local model directories: GGUF metadata parsing, safetensors, and LoRA adapters."""

from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import gguf
from app.local.hardware import estimate_model_fit, get_hardware_info

# In-memory scan cache: path -> {mtime, size, info}
_SCAN_CACHE: Dict[str, Dict[str, Any]] = {}


def read_gguf_metadata(file_path: Path) -> Dict[str, Any]:
    """Extract metadata from a GGUF file using the official gguf library."""
    stat = file_path.stat()
    size_bytes = stat.st_size
    path_str = str(file_path.resolve())

    # Check cache
    cached = _SCAN_CACHE.get(path_str)
    if cached and cached.get("mtime") == stat.st_mtime and cached.get("size") == size_bytes:
        return cached["info"]

    info: Dict[str, Any] = {
        "path": path_str,
        "filename": file_path.name,
        "format": "gguf",
        "runnable": True,
        "size_bytes": size_bytes,
        "size_formatted": f"{size_bytes / (1024 ** 3):.2f} GB",
        "architecture": "unknown",
        "name": file_path.stem,
        "quantization": "unknown",
        "context_length": 8192,
        "parameter_count": None,
        "has_chat_template": False,
        "is_mmproj": False,
    }

    if "mmproj" in file_path.name.lower():
        info["is_mmproj"] = True
        info["runnable"] = False
        info["note"] = "Multimodal Projector file"

    try:
        reader = gguf.GGUFReader(path_str)
        # Extract architecture
        arch_field = reader.get_field("general.architecture")
        if arch_field and len(arch_field.parts) > 0:
            arch = str(bytes(arch_field.parts[-1]), encoding="utf-8", errors="ignore")
            info["architecture"] = arch

            # Context length: usually <arch>.context_length
            ctx_field = reader.get_field(f"{arch}.context_length")
            if ctx_field and len(ctx_field.parts) > 0:
                info["context_length"] = int(ctx_field.parts[-1][0])

        # Model name
        name_field = reader.get_field("general.name")
        if name_field and len(name_field.parts) > 0:
            info["name"] = str(bytes(name_field.parts[-1]), encoding="utf-8", errors="ignore")

        # Quantization type
        ft_field = reader.get_field("general.file_type")
        if ft_field and len(ft_field.parts) > 0:
            file_type_code = int(ft_field.parts[-1][0])
            # Mapping common gguf file_types
            type_names = {
                0: "ALL_F32", 1: "MOSTLY_F16", 2: "MOSTLY_Q4_0", 3: "MOSTLY_Q4_1",
                7: "MOSTLY_Q8_0", 8: "MOSTLY_Q5_0", 9: "MOSTLY_Q5_1", 10: "MOSTLY_Q2_K",
                11: "MOSTLY_Q3_K_S", 12: "MOSTLY_Q3_K_M", 13: "MOSTLY_Q3_K_L",
                14: "MOSTLY_Q4_K_S", 15: "MOSTLY_Q4_K_M", 16: "MOSTLY_Q5_K_S",
                17: "MOSTLY_Q5_K_M", 18: "MOSTLY_Q6_K",
            }
            info["quantization"] = type_names.get(file_type_code, f"TYPE_{file_type_code}")

        # Chat template
        chat_template = reader.get_field("tokenizer.chat_template")
        if chat_template is not None:
            info["has_chat_template"] = True

    except Exception as e:
        info["metadata_error"] = str(e)

    _SCAN_CACHE[path_str] = {
        "mtime": stat.st_mtime,
        "size": size_bytes,
        "info": info,
    }
    return info


def scan_directory(dir_path: str, max_depth: int = 3) -> List[Dict[str, Any]]:
    """Scan directory recursively for GGUF models, safetensors, and LoRA adapters."""
    root = Path(dir_path)
    if not root.exists() or not root.is_dir():
        return []

    hw = get_hardware_info()
    models: List[Dict[str, Any]] = []
    seen_shards = set()

    for dirpath, dirnames, filenames in os.walk(root):
        current_depth = len(Path(dirpath).relative_to(root).parts)
        if current_depth > max_depth:
            dirnames.clear()
            continue

        p_dir = Path(dirpath)

        # 1. Check for Hugging Face / Unsloth safetensors model directory
        has_config = (p_dir / "config.json").exists()
        has_safetensors = any(f.endswith(".safetensors") for f in filenames)
        has_adapter_config = (p_dir / "adapter_config.json").exists()

        if has_adapter_config:
            models.append({
                "path": str(p_dir.resolve()),
                "filename": p_dir.name,
                "format": "lora_adapter",
                "runnable": False,
                "note": "LoRA Adapter (can be applied via --lora)",
            })
            dirnames.clear()  # Don't recurse deeper
            continue
        elif has_config and has_safetensors:
            models.append({
                "path": str(p_dir.resolve()),
                "filename": p_dir.name,
                "format": "safetensors",
                "runnable": False,
                "note": "Hugging Face / Unsloth weights (needs GGUF conversion for llama.cpp)",
            })
            dirnames.clear()
            continue

        # 2. Check for GGUF files
        for fname in filenames:
            if fname.lower().endswith(".gguf"):
                file_path = p_dir / fname

                # Check if it's a split shard (e.g., -00002-of-00005.gguf)
                shard_match = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", fname, re.IGNORECASE)
                if shard_match:
                    shard_num = int(shard_match.group(1))
                    total_shards = int(shard_match.group(2))
                    base_name = fname[: shard_match.start()]
                    if shard_num > 1:
                        seen_shards.add(str(file_path.resolve()))
                        continue

                info = read_gguf_metadata(file_path)
                info["fit_recommendation"] = estimate_model_fit(info["size_bytes"], hw)
                models.append(info)

    return models


def scan_granted_directories(folders: List[str]) -> List[Dict[str, Any]]:
    """Scan all user-granted directories and return deduplicated list of discovered models."""
    discovered: List[Dict[str, Any]] = []
    seen_paths = set()
    for folder in folders:
        try:
            results = scan_directory(folder)
            for m in results:
                if m["path"] not in seen_paths:
                    seen_paths.add(m["path"])
                    discovered.append(m)
        except Exception as e:
            print(f"Error scanning folder {folder}: {e}")
    return discovered

"""Folder selection helpers: native OS file dialog and server-side directory navigation."""

from __future__ import annotations
import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def pick_folder_native() -> Optional[str]:
    """Launch native OS folder picker dialog in a separate process/thread."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        folder = filedialog.askdirectory(title="Select Model Directory")
        root.destroy()
        if folder and os.path.isdir(folder):
            return str(Path(folder).resolve())
        return None
    except Exception as e:
        print(f"Native folder picker failed: {e}")
        return None


async def pick_folder_async() -> Optional[str]:
    """Run native folder dialog in thread pool to avoid blocking the event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, pick_folder_native)


def list_directory_contents(target_path: Optional[str] = None) -> Dict[str, Any]:
    """Fallback directory explorer for web GUI."""
    if not target_path or not os.path.exists(target_path):
        current = Path.home()
    else:
        current = Path(target_path).resolve()

    parent = str(current.parent) if current.parent != current else None

    folders: List[Dict[str, str]] = []
    try:
        for entry in sorted(current.iterdir(), key=lambda e: e.name.lower()):
            if entry.is_dir() and not entry.name.startswith("."):
                folders.append({"name": entry.name, "path": str(entry.resolve())})
    except PermissionError:
        pass

    # Detect logical drives on Windows
    drives: List[str] = []
    if sys.platform == "win32":
        import string
        for letter in string.ascii_uppercase:
            drive_path = f"{letter}:\\"
            if os.path.exists(drive_path):
                drives.append(drive_path)

    return {
        "current": str(current),
        "parent": parent,
        "folders": folders,
        "drives": drives,
    }

"""llama-server binary detection, process supervision, health checking, and log streaming."""

from __future__ import annotations
import atexit
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx


def find_llama_server_binary(custom_path: Optional[str] = None) -> Optional[str]:
    """Search for llama-server executable."""
    if custom_path and os.path.isfile(custom_path) and os.access(custom_path, os.X_OK):
        return str(Path(custom_path).resolve())

    # Environment variable
    env_path = os.environ.get("LLAMA_SERVER_PATH")
    if env_path and os.path.isfile(env_path):
        return str(Path(env_path).resolve())

    # PATH search
    for name in ["llama-server", "llama-server.exe"]:
        found = shutil.which(name)
        if found:
            return str(Path(found).resolve())

    # Windows common locations
    user_home = Path.home()
    localappdata = os.environ.get("LOCALAPPDATA", "")
    candidates = [
        Path(localappdata) / "llama.cpp" / "llama-server.exe" if localappdata else None,
        Path("C:/llama.cpp/llama-server.exe"),
        Path("C:/llama.cpp/build/bin/Release/llama-server.exe"),
        Path("C:/Program Files/llama.cpp/llama-server.exe"),
        user_home / "llama.cpp" / "llama-server.exe",
        user_home / "scoop" / "apps" / "llama.cpp" / "current" / "llama-server.exe",
    ]

    for c in candidates:
        if c and c.exists() and c.is_file():
            return str(c.resolve())

    return None


def get_llama_server_info(binary_path: str) -> Dict[str, Any]:
    """Inspect llama-server version and build flags."""
    try:
        proc = subprocess.run([binary_path, "--version"], capture_output=True, text=True, timeout=4)
        output = proc.stdout.strip() or proc.stderr.strip()
        return {
            "path": binary_path,
            "version": output.splitlines()[0] if output else "unknown",
            "full_output": output,
            "valid": True,
        }
    except Exception as e:
        return {
            "path": binary_path,
            "error": str(e),
            "valid": False,
        }


class LlamaServerManager:
    """Manages the lifecycle of a local llama-server subprocess."""

    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.status: str = "stopped"  # stopped, starting, running, error
        self.current_model: Optional[str] = None
        self.current_port: int = 8080
        self.logs: List[str] = []
        self.max_logs: int = 500
        self._log_thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None
        self.error_message: Optional[str] = None
        atexit.register(self.stop)

    def is_running(self) -> bool:
        if self.process and self.process.poll() is None:
            return True
        return False

    def get_status(self) -> Dict[str, Any]:
        alive = self.is_running()
        if not alive and self.status == "running":
            self.status = "stopped"
        return {
            "status": self.status,
            "is_alive": alive,
            "pid": self.process.pid if alive and self.process else None,
            "model_path": self.current_model,
            "port": self.current_port,
            "base_url": f"http://127.0.0.1:{self.current_port}/v1",
            "uptime_seconds": int(time.time() - self._start_time) if alive and self._start_time else 0,
            "recent_logs": self.logs[-50:],
            "error_message": self.error_message,
        }

    def _read_logs(self, proc: subprocess.Popen) -> None:
        """Capture stdout/stderr lines continuously into log ring buffer."""
        try:
            if proc.stdout:
                for line in iter(proc.stdout.readline, ""):
                    if line:
                        cleaned = line.rstrip()
                        self.logs.append(cleaned)
                        if len(self.logs) > self.max_logs:
                            self.logs.pop(0)
                    if proc.poll() is not None:
                        break
        except Exception:
            pass

    async def start(
        self,
        binary_path: str,
        model_path: str,
        port: int = 8080,
        ctx: int = 8192,
        ngl: int = 99,
        parallel: int = 2,
        extra_args: Optional[List[str]] = None,
    ) -> bool:
        """Start llama-server with the specified GGUF model and wait for health status."""
        self.stop()
        self.logs.clear()
        self.status = "starting"
        self.error_message = None
        self.current_model = model_path
        self.current_port = port

        cmd = [
            binary_path,
            "-m", model_path,
            "-c", str(ctx),
            "-ngl", str(ngl),
            "--parallel", str(parallel),
            "--port", str(port),
            "--host", "127.0.0.1",
            "--jinja",
        ]

        if extra_args:
            cmd.extend(extra_args)

        self.logs.append(f"[Server Manager] Starting command: {' '.join(cmd)}")

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace",
            )
            self._start_time = time.time()

            self._log_thread = threading.Thread(target=self._read_logs, args=(self.process,), daemon=True)
            self._log_thread.start()

            # Poll health endpoint
            health_url = f"http://127.0.0.1:{port}/health"
            ready = False
            for _ in range(45):  # Wait up to 45 seconds for large model loading
                await asyncio.sleep(1.0)
                if self.process.poll() is not None:
                    # Exited prematurely
                    self.status = "error"
                    self.error_message = f"Process exited with code {self.process.returncode}"
                    self.logs.append(f"[Server Manager] Error: {self.error_message}")
                    return False
                try:
                    async with httpx.AsyncClient(timeout=1.0) as client:
                        resp = await client.get(health_url)
                        if resp.status_code == 200:
                            ready = True
                            break
                except Exception:
                    pass

            if ready:
                self.status = "running"
                self.logs.append(f"[Server Manager] llama-server ready on http://127.0.0.1:{port}/v1")
                return True
            else:
                self.status = "error"
                self.error_message = "Timed out waiting for /health endpoint"
                self.logs.append(f"[Server Manager] {self.error_message}")
                return False

        except Exception as e:
            self.status = "error"
            self.error_message = str(e)
            self.logs.append(f"[Server Manager] Launch failed: {e}")
            return False

    def stop(self) -> None:
        """Stop running llama-server process."""
        if self.process:
            try:
                self.logs.append("[Server Manager] Stopping llama-server...")
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=2)
            except Exception as e:
                self.logs.append(f"[Server Manager] Stop error: {e}")
            finally:
                self.process = None
                self.status = "stopped"
                self._start_time = None


# Global singleton instance
llama_server_manager = LlamaServerManager()

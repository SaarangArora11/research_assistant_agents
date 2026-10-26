"""Configuration management and settings persistence."""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field

SETTINGS_FILE = Path("settings.json")


class Settings(BaseModel):
    # Provider selection
    provider: Literal["openrouter", "llamacpp"] = "openrouter"

    # OpenRouter
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_default_model: str = "meta-llama/llama-3.3-70b-instruct"

    # llama.cpp server
    llamacpp_base_url: str = "http://127.0.0.1:8080/v1"
    llamacpp_api_key: str = ""
    llamacpp_model_alias: str = "local-model"

    # Selected model per agent role (optional overrides, empty string uses default)
    role_models: Dict[str, str] = Field(
        default_factory=lambda: {
            "planner": "",
            "researcher": "",
            "synthesizer": "",
            "reviewer": "",
            "feedback": "",
        }
    )

    # Local model manager
    local_folders: List[str] = Field(default_factory=list)
    selected_local_model_path: Optional[str] = None
    llama_server_path: Optional[str] = None
    llama_server_ctx: int = 8192
    llama_server_ngl: int = 99
    llama_server_parallel: int = 2
    llama_server_port: int = 8080

    # Search & tools
    search_backends_order: List[str] = Field(
        default_factory=lambda: [
            "ddg",
            "searxng",
            "brave",
            "tavily",
            "serper",
            "exa",
            "wikipedia",
            "arxiv",
            "semantic_scholar",
        ]
    )
    searxng_url: str = ""
    brave_api_key: str = ""
    tavily_api_key: str = ""
    serper_api_key: str = ""
    exa_api_key: str = ""
    semantic_scholar_api_key: str = ""
    multi_search: bool = False
    search_cache_ttl_hours: int = 24

    # Execution controls
    checkpoint_mode: Literal["all", "plan_only", "autopilot"] = "all"
    max_researcher_concurrency: int = 2
    max_review_rounds: int = 2
    max_sources_per_section: int = 6
    temperature: float = 0.3


def load_settings() -> Settings:
    """Load settings from settings.json or return default settings."""
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return Settings(**data)
        except Exception as e:
            print(f"Warning: Failed to load settings.json ({e}), using defaults")
    # Also fallback to environment variables if available
    s = Settings()
    if os.environ.get("OPENROUTER_API_KEY"):
        s.openrouter_api_key = os.environ["OPENROUTER_API_KEY"]
    if os.environ.get("BRAVE_API_KEY"):
        s.brave_api_key = os.environ["BRAVE_API_KEY"]
    if os.environ.get("TAVILY_API_KEY"):
        s.tavily_api_key = os.environ["TAVILY_API_KEY"]
    return s


def save_settings(settings: Settings) -> None:
    """Save settings to settings.json."""
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings.model_dump(), f, indent=2)


def mask_secret(value: str) -> str:
    """Mask secret keys leaving only first 3 and last 4 characters."""
    if not value:
        return ""
    if len(value) <= 8:
        return "********"
    return f"{value[:3]}...{value[-4:]}"


def get_masked_settings(settings: Settings) -> dict:
    """Return dictionary with secrets masked for safe client delivery."""
    data = settings.model_dump()
    data["openrouter_api_key_masked"] = mask_secret(settings.openrouter_api_key)
    data["has_openrouter_key"] = bool(settings.openrouter_api_key)
    data["has_brave_key"] = bool(settings.brave_api_key)
    data["has_tavily_key"] = bool(settings.tavily_api_key)
    data["has_serper_key"] = bool(settings.serper_api_key)
    data["has_exa_key"] = bool(settings.exa_api_key)
    # Remove raw sensitive keys from response
    data["openrouter_api_key"] = ""
    data["brave_api_key"] = ""
    data["tavily_api_key"] = ""
    data["serper_api_key"] = ""
    data["exa_api_key"] = ""
    data["semantic_scholar_api_key"] = ""
    return data


def get_suggested_model_directories() -> List[str]:
    """Inspect standard user directories for potential GGUF/model locations."""
    candidates = []
    user_home = Path.home()

    # Hugging Face cache
    hf_hub = Path(os.environ.get("HF_HOME", user_home / ".cache" / "huggingface" / "hub"))
    if hf_hub.exists():
        candidates.append(str(hf_hub))

    # llama.cpp default cache on Windows
    localappdata = os.environ.get("LOCALAPPDATA")
    if localappdata:
        llama_cache = Path(localappdata) / "llama.cpp"
        if llama_cache.exists():
            candidates.append(str(llama_cache))

    # LM Studio default model directories
    lm_studio_1 = user_home / ".cache" / "lm-studio" / "models"
    lm_studio_2 = user_home / ".lmstudio" / "models"
    if lm_studio_1.exists():
        candidates.append(str(lm_studio_1))
    if lm_studio_2.exists():
        candidates.append(str(lm_studio_2))

    # Common manual folders
    for d in [Path("C:/models"), Path("D:/models"), user_home / "models"]:
        if d.exists():
            candidates.append(str(d))

    return candidates

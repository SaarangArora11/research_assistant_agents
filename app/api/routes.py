"""FastAPI REST API routes for settings, local models, llama-server, runs, checkpoints, and reports."""

from __future__ import annotations
import asyncio
import json
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from app.core.config import (
    Settings,
    get_masked_settings,
    get_suggested_model_directories,
    load_settings,
    save_settings,
)
from app.core.events import event_bus
from app.core.run_store import run_store
from app.llm.client import LLMClient
from app.local.folder_picker import list_directory_contents, pick_folder_async
from app.local.hardware import get_hardware_info
from app.local.llama_server import (
    find_llama_server_binary,
    get_llama_server_info,
    llama_server_manager,
)
from app.local.scanner import scan_granted_directories
from app.models import Annotation, CheckpointAction, RunState, RunSummary
from app.pipeline import ACTIVE_RUNNERS, PipelineRunner
from app.tools.search.router import SearchRouter

router = APIRouter(prefix="/api")


# -------------------------------------------------------------
# SETTINGS & HARDWARE
# -------------------------------------------------------------

@router.get("/settings")
async def get_settings_endpoint():
    s = load_settings()
    masked = get_masked_settings(s)
    hw = get_hardware_info()
    suggested = get_suggested_model_directories()
    binary_path = find_llama_server_binary(s.llama_server_path)
    binary_info = get_llama_server_info(binary_path) if binary_path else None

    return {
        "settings": masked,
        "hardware": hw,
        "suggested_directories": suggested,
        "llama_server": {
            "found": bool(binary_path),
            "path": binary_path,
            "info": binary_info,
            "status": llama_server_manager.get_status(),
        },
    }


@router.put("/settings")
async def update_settings_endpoint(new_settings: Dict[str, Any]):
    current = load_settings()
    current_data = current.model_dump()

    # Do not overwrite sensitive keys if empty string was passed back from UI
    sensitive_keys = [
        "openrouter_api_key",
        "brave_api_key",
        "tavily_api_key",
        "serper_api_key",
        "exa_api_key",
        "semantic_scholar_api_key",
    ]
    for k, v in new_settings.items():
        if k in sensitive_keys and (not v or v.strip() == ""):
            continue
        if k in current_data:
            current_data[k] = v

    updated = Settings(**current_data)
    save_settings(updated)
    return {"success": True, "settings": get_masked_settings(updated)}


@router.post("/provider/test")
async def test_provider_endpoint():
    s = load_settings()
    client = LLMClient(s)
    res = await client.test_connection()
    return res


@router.get("/provider/models")
async def get_provider_models_endpoint():
    s = load_settings()
    client = LLMClient(s)
    models = await client.list_available_models()
    return {"models": models}


# -------------------------------------------------------------
# LOCAL MODELS & LLAMA-SERVER
# -------------------------------------------------------------

@router.post("/local/folders/pick")
async def pick_folder_endpoint():
    """Trigger native folder picker dialog."""
    path = await pick_folder_async()
    return {"selected_path": path}


@router.get("/fs/list")
async def list_fs_endpoint(path: Optional[str] = Query(None)):
    """Fallback directory explorer."""
    return list_directory_contents(path)


@router.post("/local/folders")
async def add_local_folder_endpoint(payload: Dict[str, str]):
    folder = payload.get("folder", "").strip()
    if not folder:
        raise HTTPException(status_code=400, detail="Folder path is required")
    s = load_settings()
    if folder not in s.local_folders:
        s.local_folders.append(folder)
        save_settings(s)
    return {"success": True, "local_folders": s.local_folders}


@router.delete("/local/folders")
async def remove_local_folder_endpoint(payload: Dict[str, str]):
    folder = payload.get("folder", "").strip()
    s = load_settings()
    if folder in s.local_folders:
        s.local_folders.remove(folder)
        save_settings(s)
    return {"success": True, "local_folders": s.local_folders}


@router.post("/local/scan")
async def scan_local_models_endpoint():
    s = load_settings()
    models = scan_granted_directories(s.local_folders)
    return {"models": models, "count": len(models)}


@router.get("/local/llama-server")
async def get_llama_server_endpoint():
    s = load_settings()
    binary_path = find_llama_server_binary(s.llama_server_path)
    info = get_llama_server_info(binary_path) if binary_path else None
    return {
        "found": bool(binary_path),
        "path": binary_path,
        "info": info,
        "status": llama_server_manager.get_status(),
    }


class StartServerRequest(BaseModel):
    model_path: str
    port: int = 8080
    ctx: int = 8192
    ngl: int = 99
    parallel: int = 2


@router.post("/local/server/start")
async def start_server_endpoint(req: StartServerRequest):
    s = load_settings()
    binary_path = find_llama_server_binary(s.llama_server_path)
    if not binary_path:
        raise HTTPException(status_code=404, detail="llama-server executable not found on system.")

    ok = await llama_server_manager.start(
        binary_path=binary_path,
        model_path=req.model_path,
        port=req.port,
        ctx=req.ctx,
        ngl=req.ngl,
        parallel=req.parallel,
    )
    if ok:
        # Automatically update settings to point to local server
        s.llamacpp_base_url = f"http://127.0.0.1:{req.port}/v1"
        s.selected_local_model_path = req.model_path
        s.provider = "llamacpp"
        save_settings(s)

    return {"success": ok, "status": llama_server_manager.get_status()}


@router.post("/local/server/stop")
async def stop_server_endpoint():
    llama_server_manager.stop()
    return {"success": True, "status": llama_server_manager.get_status()}


@router.get("/local/server/status")
async def get_server_status_endpoint():
    return llama_server_manager.get_status()


# -------------------------------------------------------------
# SEARCH TEST
# -------------------------------------------------------------

@router.post("/search/test")
async def test_search_endpoint(payload: Dict[str, str]):
    query = payload.get("query", "artificial intelligence latest developments")
    s = load_settings()
    router = SearchRouter(s)
    results = await router.search(query, max_results=4)
    return {"query": query, "results": [r.model_dump() for r in results]}


# -------------------------------------------------------------
# RUNS & CHECKPOINTS
# -------------------------------------------------------------

class CreateRunRequest(BaseModel):
    topic: str
    extra_instructions: str = ""
    depth: str = "standard"
    checkpoint_mode: str = "all"


@router.post("/runs")
async def create_run_endpoint(req: CreateRunRequest):
    topic = req.topic.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="Topic is required")

    summary = run_store.create_run(
        topic=topic,
        extra_instructions=req.extra_instructions,
        depth=req.depth,
        checkpoint_mode=req.checkpoint_mode,
    )

    runner = PipelineRunner(summary.run_id)
    ACTIVE_RUNNERS[summary.run_id] = runner

    # Launch in background task
    asyncio.create_task(runner.execute_run())

    return summary


@router.get("/runs")
async def list_runs_endpoint():
    return {"runs": run_store.list_runs()}


@router.get("/runs/{run_id}")
async def get_run_endpoint(run_id: str):
    summary = run_store.get_run(run_id)
    if not summary:
        raise HTTPException(status_code=404, detail="Run not found")
    plan = run_store.get_plan(run_id)
    sections = run_store.get_sections(run_id)
    report = run_store.get_report(run_id)
    versions = run_store.list_report_versions(run_id)

    return {
        "summary": summary,
        "plan": plan,
        "sections": sections,
        "report": report,
        "versions": versions,
    }


@router.get("/runs/{run_id}/events")
async def stream_run_events(run_id: str):
    """Server-Sent Events endpoint for real-time agent activity."""
    async def event_generator():
        # First send recent history for replay
        history = event_bus.get_history(run_id)
        for ev in history:
            yield {"event": "agent_event", "data": json.dumps(ev.model_dump())}

        # Subscribe to new live events
        q = await event_bus.subscribe(run_id)
        try:
            while True:
                ev = await q.get()
                yield {"event": "agent_event", "data": json.dumps(ev.model_dump())}
        except asyncio.CancelledError:
            pass
        finally:
            await event_bus.unsubscribe(q, run_id)

    return EventSourceResponse(event_generator())


@router.post("/runs/{run_id}/checkpoint")
async def submit_checkpoint_endpoint(run_id: str, action: CheckpointAction):
    runner = ACTIVE_RUNNERS.get(run_id)
    if not runner:
        raise HTTPException(status_code=404, detail="Active runner not found for this run")
    runner.submit_checkpoint_action(action)
    return {"success": True}


@router.post("/runs/{run_id}/cancel")
async def cancel_run_endpoint(run_id: str):
    runner = ACTIVE_RUNNERS.get(run_id)
    if runner:
        runner.cancel()
    summary = run_store.get_run(run_id)
    if summary:
        summary.state = RunState.CANCELLED
        run_store.update_run(summary)
    return {"success": True}


# -------------------------------------------------------------
# REPORTS & ANNOTATIONS (IN-APP FEEDBACK)
# -------------------------------------------------------------

@router.get("/runs/{run_id}/report")
async def get_report_endpoint(run_id: str, version: Optional[int] = Query(None)):
    report = run_store.get_report(run_id, version=version)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    versions = run_store.list_report_versions(run_id)
    return {"version": version or (versions[-1] if versions else 1), "markdown": report, "versions": versions}


@router.get("/runs/{run_id}/annotations")
async def get_annotations_endpoint(run_id: str, version: int = Query(1)):
    annotations = run_store.get_annotations(run_id, version=version)
    return {"annotations": [a.model_dump() for a in annotations]}


class SaveAnnotationsRequest(BaseModel):
    version: int
    annotations: List[Annotation]


@router.post("/runs/{run_id}/annotations")
async def save_annotations_endpoint(run_id: str, req: SaveAnnotationsRequest):
    run_store.save_annotations(run_id, req.version, req.annotations)
    return {"success": True, "count": len(req.annotations)}


@router.post("/runs/{run_id}/feedback")
async def trigger_feedback_revision_endpoint(run_id: str, req: SaveAnnotationsRequest):
    """Feed annotations to Feedback Triage agent and produce a new versioned report."""
    runner = ACTIVE_RUNNERS.get(run_id)
    if not runner:
        runner = PipelineRunner(run_id)
        ACTIVE_RUNNERS[run_id] = runner

    # Launch revision in background
    asyncio.create_task(runner.execute_annotations_feedback(req.annotations))
    return {"success": True, "message": "Feedback revision triggered"}

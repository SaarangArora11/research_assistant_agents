"""FastAPI main application entrypoint."""

from __future__ import annotations
import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from app.api.routes import router as api_router

app = FastAPI(
    title="Multi-Agent Research Assistant",
    description="Collaborative multi-agent research app with local llama.cpp and OpenRouter support.",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Router
app.include_router(api_router)

# Mount Web Directory
web_dir = Path("web").resolve()
web_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")


@app.get("/")
async def root():
    index_file = web_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Multi-Agent Research Assistant API is running. UI is being prepared."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)

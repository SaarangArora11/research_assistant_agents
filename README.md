# SynapseResearch — Multi-Agent Deep Research Assistant

A collaborative multi-agent research application where an **Orchestrator** creates a research plan, **Researcher subagents** investigate sub-topics concurrently using live web tools, a **Synthesizer agent** harmonizes findings into a single cited report, and an in-app **Review & Annotation system** allows you to highlight text and steer iterative re-research.

Supports both **Local Models (llama.cpp)** and **Cloud Models (OpenRouter)** with a dark-themed glassmorphism GUI.

---

## Key Features

1. **Multi-Agent Orchestration & Steerability**:
   - **Orchestrator (Planner)**: Performs scoping web searches, constructs a structured research plan with MECE sub-tasks.
   - **Researchers (×N)**: Search web backends, fetch pages, parse PDFs, extract evidence, and draft section reports with bracketed citations (`[S1]`, `[S2]`).
   - **Synthesizer**: Harmonizes citations across all sections into a unified global bibliography and merges sections into an executive report.
   - **Orchestrator (Reviewer)**: Fact-checks key claims on the live web and critiques coverage against the plan in an iterative loop.
   - **Feedback Triage Agent**: Translates in-app user highlights and comments into targeted sub-tasks and revisions.

2. **3 Human-in-the-Loop Checkpoints**:
   - **Checkpoint 1 (Plan Review)**: Review, edit, or provide steering feedback on sub-tasks before research begins.
   - **Checkpoint 2 (Sections Review)**: Inspect individual section reports and sources before synthesis.
   - **Checkpoint 3 (Draft Review)**: Review full synthesized draft and fact-check findings.
   - *Modes*: **All Stops (Recommended)**, **Plan Only**, or **Autopilot**.

3. **In-App Highlight & Comment Feedback Loop**:
   - Select any text passage in the rendered report to pop up **💬 Comment & Steer**.
   - Choose intent: *Research More*, *Fact-Check*, *Rewrite*, *Expand*, or *Remove*.
   - Click **Generate Revised Report** to dispatch targeted subagents and create versioned reports (`v1`, `v2`, `v3`...).

4. **Web Search & Fetch with Fallback Chain**:
   - **Search Engines**: DuckDuckGo, SearXNG, Brave, Tavily, Serper (Google), Exa, Wikipedia, arXiv, and Semantic Scholar.
   - **Automatic Fallback**: If an engine fails, hits rate limits, or returns empty results, the router immediately fails over to the next engine in the priority chain with circuit breaker cooldowns.
   - **Content Extraction**: `httpx` + `trafilatura` for clean article text, `pypdf` for scientific papers, and Jina Reader fallback.

5. **Local Model Manager (llama.cpp) & OpenRouter**:
   - **Directory Scanner**: Grants access to local folders (Hugging Face cache, LM Studio, custom folders). Reads GGUF metadata (architecture, quantization, context length).
   - **Hardware Fit Estimator**: Automatically detects GPU VRAM and System RAM to recommend GPU layer offload (`-ngl`) and context length (`-c`).
   - **llama-server Supervision**: Starts, monitors health (`/health`), and streams server logs directly in the GUI.
   - **OpenRouter**: Toggle to cloud models (e.g. Claude 3.5, Llama 3.3 70B, DeepSeek) with a single API key.

---

## Quickstart

### 1. Requirements
- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) package manager

### 2. Installation
The local virtual environment and dependencies are managed via `uv`:

```bash
# Run the FastAPI server
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open your browser to:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

### 3. Running Tests
To run the automated test suite:
```bash
uv run pytest -v
```

---

## Configuration & Settings

All settings are stored locally in `settings.json` (git-ignored) and can be configured directly from the in-app **Settings** tab:
- **LLM Provider**: Toggle between `openrouter` and `llamacpp`.
- **OpenRouter API Key**: Set your key (masked in UI).
- **llama.cpp URL**: Default `http://127.0.0.1:8080/v1`.
- **Search Backends**: Reorder fallback priority and add optional API keys for Brave, Tavily, or Serper.

---

## Project Structure

```
research_assistant_agents/
├── app/
│   ├── main.py                  # FastAPI application and route mounting
│   ├── models.py                # Core Pydantic data schemas
│   ├── pipeline.py              # Orchestrator of agent pipeline and checkpoints
│   ├── api/
│   │   └── routes.py            # REST API & Server-Sent Events (SSE)
│   ├── core/
│   │   ├── config.py            # Settings persistence & secret masking
│   │   ├── events.py            # Async pub-sub event bus
│   │   └── run_store.py         # File-based persistence of runs, reports, and annotations
│   ├── llm/
│   │   └── client.py            # Unified async client for llama.cpp & OpenRouter
│   ├── local/
│   │   ├── scanner.py           # GGUF metadata parsing & model discovery
│   │   ├── hardware.py          # RAM/VRAM inspection & fit estimates
│   │   ├── llama_server.py      # llama-server process lifecycle management
│   │   └── folder_picker.py     # Native OS folder dialog & directory browser
│   ├── tools/
│   │   ├── search/              # Search engines (DDG, Wikipedia, arXiv, etc.) & router
│   │   └── fetch/               # Page fetcher with trafilatura, PDF parser, Jina fallback
│   └── agents/
│       ├── orchestrator.py      # Scoping, planning, and draft review with fact-checks
│       ├── researcher.py        # Sub-topic investigation and section reporting
│       ├── synthesizer.py       # Citation harmonization and final report synthesis
│       └── feedback_triage.py   # User annotation interpretation & revision planning
├── web/
│   ├── index.html               # Single-page web interface
│   ├── styles.css               # Glassmorphic dark styling & responsive layouts
│   └── app.js                   # UI controller, SSE listener & annotation engine
├── tests/                       # Pytest test suite
├── runs/                        # Run storage (git-ignored)
└── pyproject.toml               # Project metadata & uv dependencies
```

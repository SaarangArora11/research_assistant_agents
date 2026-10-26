"""Disk persistence for research runs, plans, section reports, and versioned final reports."""

from __future__ import annotations
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from app.models import (
    Annotation,
    CheckpointAction,
    Critique,
    ResearchPlan,
    RunState,
    RunSummary,
    SectionReport,
)

RUNS_DIR = Path("runs")


def _slugify(text: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", text).strip().lower()
    slug = re.sub(r"[-\s]+", "_", slug)
    return slug[:40] if slug else "run"


class RunStore:
    def __init__(self, base_dir: Path = RUNS_DIR):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_run_dir(self, run_id: str) -> Path:
        # Search for directory matching run_id or ending with run_id
        for d in self.base_dir.iterdir():
            if d.is_dir() and (d.name == run_id or d.name.endswith(f"_{run_id}")):
                return d
        # Default path
        return self.base_dir / run_id

    def create_run(
        self,
        topic: str,
        extra_instructions: str = "",
        depth: str = "standard",
        checkpoint_mode: str = "all",
    ) -> RunSummary:
        """Create a new run folder and initialize run.json."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        short_id = os.urandom(3).hex()
        slug = _slugify(topic)
        folder_name = f"{timestamp}_{slug}_{short_id}"
        run_dir = self.base_dir / folder_name
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "sections").mkdir(exist_ok=True)
        (run_dir / "sources").mkdir(exist_ok=True)
        (run_dir / "drafts").mkdir(exist_ok=True)
        (run_dir / "reports").mkdir(exist_ok=True)
        (run_dir / "annotations").mkdir(exist_ok=True)

        summary = RunSummary(
            run_id=short_id,
            topic=topic,
            extra_instructions=extra_instructions,
            depth=depth,  # type: ignore
            checkpoint_mode=checkpoint_mode,  # type: ignore
            state=RunState.INITIALIZING,
        )

        with open(run_dir / "run.json", "w", encoding="utf-8") as f:
            json.dump(summary.model_dump(), f, indent=2)

        return summary

    def get_run(self, run_id: str) -> Optional[RunSummary]:
        run_dir = self._get_run_dir(run_id)
        run_json = run_dir / "run.json"
        if not run_json.exists():
            return None
        try:
            with open(run_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            return RunSummary(**data)
        except Exception:
            return None

    def update_run(self, summary: RunSummary) -> None:
        summary.updated_at = time.time()
        run_dir = self._get_run_dir(summary.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "run.json", "w", encoding="utf-8") as f:
            json.dump(summary.model_dump(), f, indent=2)

    def list_runs(self) -> List[RunSummary]:
        runs: List[RunSummary] = []
        if not self.base_dir.exists():
            return runs
        for d in sorted(self.base_dir.iterdir(), reverse=True):
            if d.is_dir() and (d / "run.json").exists():
                try:
                    with open(d / "run.json", "r", encoding="utf-8") as f:
                        data = json.load(f)
                    runs.append(RunSummary(**data))
                except Exception:
                    pass
        return runs

    def save_plan(self, run_id: str, plan: ResearchPlan, version: int = 1) -> None:
        run_dir = self._get_run_dir(run_id)
        plan_path = run_dir / f"plan_v{version}.json"
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(plan.model_dump(), f, indent=2)
        # Also maintain latest plan.json
        with open(run_dir / "plan.json", "w", encoding="utf-8") as f:
            json.dump(plan.model_dump(), f, indent=2)

    def get_plan(self, run_id: str, version: Optional[int] = None) -> Optional[ResearchPlan]:
        run_dir = self._get_run_dir(run_id)
        target = run_dir / f"plan_v{version}.json" if version else run_dir / "plan.json"
        if not target.exists():
            return None
        try:
            with open(target, "r", encoding="utf-8") as f:
                return ResearchPlan(**json.load(f))
        except Exception:
            return None

    def save_section(self, run_id: str, section: SectionReport, version: int = 1) -> None:
        run_dir = self._get_run_dir(run_id)
        sections_dir = run_dir / "sections"
        sections_dir.mkdir(exist_ok=True)
        file_base = f"{section.subtask_id}_v{version}"
        with open(sections_dir / f"{file_base}.json", "w", encoding="utf-8") as f:
            json.dump(section.model_dump(), f, indent=2)
        with open(sections_dir / f"{file_base}.md", "w", encoding="utf-8") as f:
            f.write(section.markdown)

    def get_sections(self, run_id: str) -> List[SectionReport]:
        run_dir = self._get_run_dir(run_id)
        sections_dir = run_dir / "sections"
        sections_dict: Dict[str, SectionReport] = {}
        if not sections_dir.exists():
            return []
        # Get highest version for each subtask
        for p in sorted(sections_dir.glob("*.json")):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    sec = SectionReport(**json.load(f))
                    sections_dict[sec.subtask_id] = sec
            except Exception:
                pass
        return list(sections_dict.values())

    def save_draft(self, run_id: str, draft: str, round_num: int) -> None:
        run_dir = self._get_run_dir(run_id)
        drafts_dir = run_dir / "drafts"
        drafts_dir.mkdir(exist_ok=True)
        with open(drafts_dir / f"draft_{round_num}.md", "w", encoding="utf-8") as f:
            f.write(draft)

    def save_critique(self, run_id: str, critique: Critique, round_num: int) -> None:
        run_dir = self._get_run_dir(run_id)
        drafts_dir = run_dir / "drafts"
        drafts_dir.mkdir(exist_ok=True)
        with open(drafts_dir / f"critique_{round_num}.json", "w", encoding="utf-8") as f:
            json.dump(critique.model_dump(), f, indent=2)

    def save_report(self, run_id: str, markdown: str, version: int = 1) -> None:
        run_dir = self._get_run_dir(run_id)
        reports_dir = run_dir / "reports"
        reports_dir.mkdir(exist_ok=True)
        with open(reports_dir / f"report_v{version}.md", "w", encoding="utf-8") as f:
            f.write(markdown)
        # Update latest final_report.md
        with open(run_dir / "final_report.md", "w", encoding="utf-8") as f:
            f.write(markdown)

    def get_report(self, run_id: str, version: Optional[int] = None) -> Optional[str]:
        run_dir = self._get_run_dir(run_id)
        if version:
            target = run_dir / "reports" / f"report_v{version}.md"
        else:
            target = run_dir / "final_report.md"
        if not target.exists():
            return None
        with open(target, "r", encoding="utf-8") as f:
            return f.read()

    def list_report_versions(self, run_id: str) -> List[int]:
        run_dir = self._get_run_dir(run_id)
        reports_dir = run_dir / "reports"
        versions = []
        if reports_dir.exists():
            for p in reports_dir.glob("report_v*.md"):
                match = re.search(r"report_v(\d+)\.md", p.name)
                if match:
                    versions.append(int(match.group(1)))
        return sorted(versions) if versions else ([1] if (run_dir / "final_report.md").exists() else [])

    def save_annotations(self, run_id: str, version: int, annotations: List[Annotation]) -> None:
        run_dir = self._get_run_dir(run_id)
        annot_dir = run_dir / "annotations"
        annot_dir.mkdir(exist_ok=True)
        data = [a.model_dump() for a in annotations]
        with open(annot_dir / f"annotations_v{version}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get_annotations(self, run_id: str, version: int) -> List[Annotation]:
        run_dir = self._get_run_dir(run_id)
        annot_file = run_dir / "annotations" / f"annotations_v{version}.json"
        if not annot_file.exists():
            return []
        try:
            with open(annot_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [Annotation(**item) for item in data]
        except Exception:
            return []

    def save_checkpoint_log(self, run_id: str, action: CheckpointAction) -> None:
        run_dir = self._get_run_dir(run_id)
        log_file = run_dir / "checkpoints.json"
        logs = []
        if log_file.exists():
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    logs = json.load(f)
            except Exception:
                logs = []
        logs.append({
            "timestamp": time.time(),
            "stage": action.stage,
            "action": action.action,
            "feedback_text": action.feedback_text,
            "edited_data": action.edited_data,
        })
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2)


# Global singleton instance
run_store = RunStore()

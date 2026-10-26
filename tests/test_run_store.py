"""Tests for run store persistence."""

from pathlib import Path
import tempfile
from app.core.run_store import RunStore
from app.models import (
    Annotation,
    AnnotationIntent,
    ResearchPlan,
    RunState,
    SectionReport,
    Source,
    SubTask,
)


def test_run_store_lifecycle():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = RunStore(base_dir=Path(tmpdir))

        # Create run
        summary = store.create_run(topic="Quantum Machine Learning", depth="quick")
        assert summary.run_id is not None
        assert summary.state == RunState.INITIALIZING

        # Save and get plan
        plan = ResearchPlan(
            topic="Quantum Machine Learning",
            interpretation="Exploring NISQ algorithms",
            audience="Researchers",
            subtasks=[SubTask(id="st_1", title="VQE", objective="Varational Quantum Eigensolver")],
        )
        store.save_plan(summary.run_id, plan, version=1)
        loaded_plan = store.get_plan(summary.run_id)
        assert loaded_plan is not None
        assert loaded_plan.topic == "Quantum Machine Learning"

        # Save section
        sec = SectionReport(
            subtask_id="st_1",
            title="VQE",
            markdown="VQE is an algorithm [S1].",
            sources=[Source(id="S1", url="https://arxiv.org/abs/1304.3061", title="VQE Paper", snippet="VQE ansatz")],
        )
        store.save_section(summary.run_id, sec, version=1)
        sections = store.get_sections(summary.run_id)
        assert len(sections) == 1
        assert sections[0].subtask_id == "st_1"

        # Save report and versions
        store.save_report(summary.run_id, "# Final Report v1", version=1)
        store.save_report(summary.run_id, "# Final Report v2 with more details", version=2)
        versions = store.list_report_versions(summary.run_id)
        assert 1 in versions
        assert 2 in versions
        assert "v2" in store.get_report(summary.run_id, version=2)

        # Save annotations
        annot = Annotation(
            report_version=1,
            quote="VQE is an algorithm",
            comment="Cite Peruzzo et al.",
            intent=AnnotationIntent.RESEARCH_MORE,
        )
        store.save_annotations(summary.run_id, 1, [annot])
        loaded_annots = store.get_annotations(summary.run_id, 1)
        assert len(loaded_annots) == 1
        assert loaded_annots[0].comment == "Cite Peruzzo et al."

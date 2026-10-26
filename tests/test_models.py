"""Tests for data models."""

from app.models import (
    Annotation,
    AnnotationIntent,
    CheckpointAction,
    Critique,
    FactCheckItem,
    ResearchPlan,
    RunState,
    SectionReport,
    Source,
    SubTask,
)


def test_research_plan_schema():
    subtask = SubTask(
        id="subtask_1",
        title="Cathode Chemistry Innovations",
        objective="Analyze latest cathode compositions",
        key_questions=["What is the energy density?", "What are degradation rates?"],
        suggested_queries=["solid-state battery cathode 2026"],
        priority=1,
    )
    plan = ResearchPlan(
        topic="Solid State Batteries",
        interpretation="Comprehensive landscape of solid state chemistry",
        audience="Technical",
        subtasks=[subtask],
        final_report_outline=["Executive Summary", "Cathode Chemistries", "Conclusion"],
    )
    assert len(plan.subtasks) == 1
    assert plan.subtasks[0].id == "subtask_1"
    dump = plan.model_dump()
    assert dump["topic"] == "Solid State Batteries"


def test_section_report_schema():
    src = Source(id="S1", url="https://example.com/battery", title="Battery Lab", snippet="High energy density")
    section = SectionReport(
        subtask_id="subtask_1",
        title="Cathode Chemistry",
        markdown="The cathode achieves 450 Wh/kg [S1].",
        key_findings=["450 Wh/kg achieved"],
        sources=[src],
        confidence="high",
    )
    assert section.confidence == "high"
    assert len(section.sources) == 1
    assert "[S1]" in section.markdown


def test_critique_schema():
    critique = Critique(
        approved=True,
        issues=[],
        missing_coverage=[],
        fact_checks=[FactCheckItem(claim="450 Wh/kg", verdict="verified")],
        instructions_for_revision="",
    )
    assert critique.approved is True
    assert len(critique.fact_checks) == 1


def test_annotation_schema():
    annot = Annotation(
        report_version=1,
        quote="450 Wh/kg achieved",
        comment="Double check if this is cell or pack level",
        intent=AnnotationIntent.FACT_CHECK,
    )
    assert annot.intent == AnnotationIntent.FACT_CHECK
    assert annot.id is not None

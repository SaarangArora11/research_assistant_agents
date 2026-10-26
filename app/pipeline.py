"""End-to-end multi-agent pipeline orchestrator with human-in-the-loop checkpoints."""

from __future__ import annotations
import asyncio
import time
from typing import Dict, List, Optional
from app.agents.feedback_triage import FeedbackTriageAgent
from app.agents.orchestrator import OrchestratorAgent
from app.agents.researcher import ResearcherAgent
from app.agents.synthesizer import SynthesizerAgent
from app.core.config import Settings, load_settings
from app.core.events import event_bus
from app.core.run_store import run_store
from app.llm.client import LLMClient
from app.models import (
    AgentEvent,
    Annotation,
    CheckpointAction,
    Critique,
    ResearchPlan,
    RunState,
    RunSummary,
    SectionReport,
    SubTask,
)
from app.tools.fetch.router import fetch_router
from app.tools.search.router import SearchRouter


class PipelineRunner:
    """Coordinates the execution of agents, checkpoints, and revision cycles for a run."""

    def __init__(self, run_id: str, settings: Optional[Settings] = None):
        self.run_id = run_id
        self.settings = settings or load_settings()
        self.search_router = SearchRouter(self.settings)
        self.llm = LLMClient(self.settings)

        self.orchestrator = OrchestratorAgent(self.llm, self.search_router)
        self.researcher = ResearcherAgent(self.llm, self.search_router, fetch_router)
        self.synthesizer = SynthesizerAgent(self.llm)
        self.feedback_triage = FeedbackTriageAgent(self.llm)

        self._checkpoint_events: Dict[str, asyncio.Event] = {}
        self._checkpoint_responses: Dict[str, CheckpointAction] = {}
        self._cancel_requested = False

    def cancel(self) -> None:
        """Signal cancellation for this run."""
        self._cancel_requested = True
        for ev in self._checkpoint_events.values():
            ev.set()

    def submit_checkpoint_action(self, action: CheckpointAction) -> None:
        """User submitted an action for a paused checkpoint."""
        stage = action.stage
        self._checkpoint_responses[stage] = action
        if stage in self._checkpoint_events:
            self._checkpoint_events[stage].set()

    async def _wait_for_checkpoint(self, stage: str, state_during_pause: RunState) -> CheckpointAction:
        """Pause pipeline execution until user provides feedback or approval."""
        summary = run_store.get_run(self.run_id)
        if summary:
            summary.state = state_during_pause
            run_store.update_run(summary)

        ev = asyncio.Event()
        self._checkpoint_events[stage] = ev

        await event_bus.publish(
            AgentEvent(
                run_id=self.run_id,
                stage=stage,
                agent_id="system",
                agent_name="System",
                status="progress",
                message=f"Pipeline paused at Checkpoint: '{stage}'. Awaiting your review.",
            )
        )

        await ev.wait()

        if self._cancel_requested:
            raise asyncio.CancelledError("Run was cancelled by user")

        action = self._checkpoint_responses.get(stage, CheckpointAction(stage=stage, action="approve"))  # type: ignore
        run_store.save_checkpoint_log(self.run_id, action)
        return action

    async def execute_run(self) -> None:
        """Execute the full research pipeline."""
        summary = run_store.get_run(self.run_id)
        if not summary:
            return

        try:
            # ---------------------------------------------------------
            # 1. PLANNING PHASE
            # ---------------------------------------------------------
            summary.state = RunState.PLANNING
            run_store.update_run(summary)

            plan = await self.orchestrator.create_plan(
                topic=summary.topic,
                extra_instructions=summary.extra_instructions,
                depth=summary.depth,
                run_id=self.run_id,
            )
            run_store.save_plan(self.run_id, plan, version=1)

            # ---------------------------------------------------------
            # CHECKPOINT 1: PLAN REVIEW
            # ---------------------------------------------------------
            if summary.checkpoint_mode in ("all", "plan_only"):
                plan_approved = False
                while not plan_approved:
                    action = await self._wait_for_checkpoint("plan", RunState.AWAITING_PLAN_REVIEW)
                    if action.action == "approve":
                        plan_approved = True
                    elif action.action == "edit" and action.edited_data:
                        # Update plan with user edits
                        try:
                            plan = ResearchPlan(**action.edited_data)
                            run_store.save_plan(self.run_id, plan, version=1)
                        except Exception as e:
                            print(f"Failed to parse edited plan: {e}")
                        plan_approved = True
                    elif action.action == "feedback" and action.feedback_text:
                        # Re-run planner with feedback
                        summary.state = RunState.PLANNING
                        run_store.update_run(summary)
                        feedback_instruction = f"{summary.extra_instructions}\nUSER FEEDBACK ON PREVIOUS PLAN: {action.feedback_text}"
                        plan = await self.orchestrator.create_plan(
                            topic=summary.topic,
                            extra_instructions=feedback_instruction,
                            depth=summary.depth,
                            run_id=self.run_id,
                        )
                        run_store.save_plan(self.run_id, plan, version=2)

            # ---------------------------------------------------------
            # 2. RESEARCH PHASE (Parallel Subagents)
            # ---------------------------------------------------------
            summary.state = RunState.RESEARCHING
            run_store.update_run(summary)

            concurrency = max(1, self.settings.max_researcher_concurrency)
            semaphore = asyncio.Semaphore(concurrency)
            max_sources = self.settings.max_sources_per_section

            async def run_subtask_with_sem(st: SubTask) -> SectionReport:
                async with semaphore:
                    if self._cancel_requested:
                        raise asyncio.CancelledError()
                    report = await self.researcher.research(
                        subtask=st,
                        overall_topic=summary.topic,
                        max_sources=max_sources,
                        run_id=self.run_id,
                    )
                    run_store.save_section(self.run_id, report, version=1)
                    return report

            section_tasks = [run_subtask_with_sem(st) for st in plan.subtasks]
            sections = await asyncio.gather(*section_tasks)

            # Update total sources in summary
            all_source_urls = {src.url for sec in sections for src in sec.sources}
            summary.total_sources = len(all_source_urls)
            run_store.update_run(summary)

            # ---------------------------------------------------------
            # CHECKPOINT 2: SECTIONS REVIEW
            # ---------------------------------------------------------
            if summary.checkpoint_mode == "all":
                action = await self._wait_for_checkpoint("sections", RunState.AWAITING_SECTIONS_REVIEW)
                if action.action == "feedback" and action.feedback_text:
                    # User requested adjustments to sections
                    await event_bus.publish(
                        AgentEvent(
                            run_id=self.run_id,
                            stage="sections",
                            agent_id="system",
                            agent_name="System",
                            status="progress",
                            message=f"Received feedback on sections: '{action.feedback_text}'. Incorporating into synthesis.",
                        )
                    )

            # ---------------------------------------------------------
            # 3. SYNTHESIS & REVIEW LOOP
            # ---------------------------------------------------------
            summary.state = RunState.SYNTHESIZING
            run_store.update_run(summary)

            draft = await self.synthesizer.synthesize(
                plan=plan,
                sections=sections,
                run_id=self.run_id,
            )
            run_store.save_draft(self.run_id, draft, round_num=1)

            # Review loop
            for r in range(1, self.settings.max_review_rounds + 1):
                summary.state = RunState.REVIEWING
                run_store.update_run(summary)

                critique = await self.orchestrator.review_draft(
                    plan=plan,
                    draft_markdown=draft,
                    sections=sections,
                    run_id=self.run_id,
                )
                run_store.save_critique(self.run_id, critique, round_num=r)

                if critique.approved or r == self.settings.max_review_rounds:
                    break

                # Revise draft
                summary.state = RunState.SYNTHESIZING
                run_store.update_run(summary)
                draft = await self.synthesizer.synthesize(
                    plan=plan,
                    sections=sections,
                    previous_draft=draft,
                    critique=critique,
                    run_id=self.run_id,
                )
                run_store.save_draft(self.run_id, draft, round_num=r + 1)

            # ---------------------------------------------------------
            # CHECKPOINT 3: DRAFT REVIEW
            # ---------------------------------------------------------
            if summary.checkpoint_mode == "all":
                action = await self._wait_for_checkpoint("draft", RunState.AWAITING_DRAFT_REVIEW)
                if action.action == "feedback" and action.feedback_text:
                    # Final user revision pass
                    draft = await self.synthesizer.synthesize(
                        plan=plan,
                        sections=sections,
                        previous_draft=draft,
                        user_feedback=action.feedback_text,
                        run_id=self.run_id,
                    )
                elif action.action == "edit" and action.edited_data and "markdown" in action.edited_data:
                    draft = action.edited_data["markdown"]

            # ---------------------------------------------------------
            # 4. FINAL REPORT
            # ---------------------------------------------------------
            run_store.save_report(self.run_id, draft, version=1)

            summary.state = RunState.DONE
            summary.current_version = 1
            summary.completed_at = time.time()
            run_store.update_run(summary)

            await event_bus.publish(
                AgentEvent(
                    run_id=self.run_id,
                    stage="done",
                    agent_id="system",
                    agent_name="System",
                    status="completed",
                    message="Research report generated successfully! Ready for in-app review.",
                    data={"version": 1},
                )
            )

        except asyncio.CancelledError:
            summary.state = RunState.CANCELLED
            run_store.update_run(summary)
            await event_bus.publish(
                AgentEvent(
                    run_id=self.run_id,
                    stage="cancelled",
                    agent_id="system",
                    agent_name="System",
                    status="warning",
                    message="Research run was cancelled.",
                )
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            summary.state = RunState.FAILED
            summary.error_message = str(e)
            run_store.update_run(summary)
            await event_bus.publish(
                AgentEvent(
                    run_id=self.run_id,
                    stage="error",
                    agent_id="system",
                    agent_name="System",
                    status="error",
                    message=f"Research pipeline encountered an error: {e}",
                )
            )

    async def execute_annotations_feedback(self, annotations: List[Annotation]) -> int:
        """Process user highlights/comments to perform targeted re-research and output a new report version."""
        summary = run_store.get_run(self.run_id)
        if not summary:
            return 1

        curr_version = summary.current_version
        next_version = curr_version + 1

        summary.state = RunState.REVISING
        run_store.update_run(summary)

        current_report = run_store.get_report(self.run_id, version=curr_version) or ""
        plan = run_store.get_plan(self.run_id) or ResearchPlan(topic=summary.topic, interpretation="", audience="")
        existing_sections = run_store.get_sections(self.run_id)

        # 1. Triage annotations
        feedback_plan = await self.feedback_triage.triage(
            report_markdown=current_report,
            annotations=annotations,
            run_id=self.run_id,
        )

        # 2. Run new targeted subtasks if needed
        new_sections: List[SectionReport] = []
        if feedback_plan.subtasks_to_add:
            for st in feedback_plan.subtasks_to_add:
                sec = await self.researcher.research(
                    subtask=st,
                    overall_topic=summary.topic,
                    max_sources=self.settings.max_sources_per_section,
                    run_id=self.run_id,
                )
                run_store.save_section(self.run_id, sec, version=next_version)
                new_sections.append(sec)

        all_sections = existing_sections + new_sections

        # 3. Synthesize revised report
        combined_feedback = (
            f"REVISION INSTRUCTIONS BASED ON USER IN-APP ANNOTATIONS:\n{feedback_plan.rewrite_instructions}\n\n"
            f"USER ANNOTATION HIGHLIGHTS:\n"
            + "\n".join([f'- Quoted "{a.quote}": {a.comment}' for a in annotations])
        )

        revised_report = await self.synthesizer.synthesize(
            plan=plan,
            sections=all_sections,
            previous_draft=current_report,
            user_feedback=combined_feedback,
            run_id=self.run_id,
        )

        # 4. Save versioned report
        run_store.save_report(self.run_id, revised_report, version=next_version)
        run_store.save_annotations(self.run_id, curr_version, annotations)

        summary.current_version = next_version
        summary.state = RunState.DONE
        run_store.update_run(summary)

        await event_bus.publish(
            AgentEvent(
                run_id=self.run_id,
                stage="done",
                agent_id="system",
                agent_name="System",
                status="completed",
                message=f"Revised Report Version {next_version} generated successfully!",
                data={"version": next_version},
            )
        )

        return next_version


# Active runners dictionary: run_id -> PipelineRunner
ACTIVE_RUNNERS: Dict[str, PipelineRunner] = {}

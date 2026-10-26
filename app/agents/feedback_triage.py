"""Feedback Triage Agent: Analyzes user highlights and comments to orchestrate targeted re-research and report revision."""

from __future__ import annotations
from typing import List
from app.core.events import event_bus
from app.llm.client import LLMClient
from app.models import (
    AgentEvent,
    Annotation,
    FeedbackPlan,
    SubTask,
)


class FeedbackTriageAgent:
    """Interprets in-app user annotations (highlights and comments) and formulates an action plan."""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    async def triage(
        self,
        report_markdown: str,
        annotations: List[Annotation],
        run_id: str = "",
    ) -> FeedbackPlan:
        await event_bus.publish(
            AgentEvent(
                run_id=run_id,
                stage="revising",
                agent_id="feedback_triage",
                agent_name="Feedback Triage Agent",
                status="started",
                message=f"Analyzing {len(annotations)} user feedback annotations...",
            )
        )

        # Format annotations context
        annot_context = []
        for idx, a in enumerate(annotations):
            annot_context.append(
                f"Annotation #{idx+1} [ID: {a.id}] (Intent: {a.intent.value}):\n"
                f'Highlighted Quote: "{a.quote}"\n'
                f'User Comment: "{a.comment}"\n'
            )
        annotations_text = "\n".join(annot_context)

        prompt = (
            f"You are the Feedback Triage Specialist for the research assistant.\n"
            f"The user reviewed the generated research report and highlighted specific passages with actionable comments:\n\n"
            f"{annotations_text}\n\n"
            f"REPORT EXCERPT FOR CONTEXT:\n{report_markdown[:4000]}\n\n"
            f"Your goal is to formulate an actionable plan to address all user comments.\n"
            f"1. For comments requesting more depth, missing data, or new areas, define new targeted SubTasks to investigate.\n"
            f"2. For comments asking to rewrite, rephrase, or remove sections, generate clear editorial instructions.\n\n"
            f"Return a JSON object conforming to the FeedbackPlan schema:\n"
            f"- actions: array of objects with (annotation_id, action, details, suggested_query)\n"
            f"- subtasks_to_add: array of SubTask objects (id, title, objective, key_questions, suggested_queries, priority)\n"
            f"- rewrite_instructions: string (summary of stylistic/content adjustments needed)"
        )

        response = await self.llm.chat(
            messages=[
                {"role": "system", "content": "You are an analytical feedback triage specialist. Output valid JSON."},
                {"role": "user", "content": prompt},
            ],
            role="feedback",
            response_schema=FeedbackPlan,
        )

        plan: FeedbackPlan = response.parsed  # type: ignore

        await event_bus.publish(
            AgentEvent(
                run_id=run_id,
                stage="revising",
                agent_id="feedback_triage",
                agent_name="Feedback Triage Agent",
                status="completed",
                message=f"Triage complete: {len(plan.subtasks_to_add)} new research subtasks created, along with editorial instructions.",
                data={"actions_count": len(plan.actions), "new_subtasks": len(plan.subtasks_to_add)},
            )
        )

        return plan

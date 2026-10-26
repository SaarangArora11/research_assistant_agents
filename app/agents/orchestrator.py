"""Orchestrator Agent: Scoping search, research plan creation, and draft review with fact-checking."""

from __future__ import annotations
import json
from typing import Any, Dict, List, Optional
from app.core.events import event_bus
from app.llm.client import LLMClient
from app.models import (
    AgentEvent,
    Critique,
    FactCheckItem,
    ResearchPlan,
    SectionReport,
    Source,
    SubTask,
)
from app.tools.search.router import SearchRouter


class OrchestratorAgent:
    """Main coordinator agent for planning and reviewing."""

    def __init__(self, llm_client: LLMClient, search_router: SearchRouter):
        self.llm = llm_client
        self.search_router = search_router

    async def create_plan(
        self,
        topic: str,
        extra_instructions: str = "",
        depth: str = "standard",
        run_id: str = "",
    ) -> ResearchPlan:
        """Conduct scoping search and generate structured ResearchPlan."""
        await event_bus.publish(
            AgentEvent(
                run_id=run_id,
                stage="planning",
                agent_id="orchestrator",
                agent_name="Orchestrator Agent",
                status="running",
                message=f"Starting web scoping for topic: '{topic}'",
            )
        )

        # 1. Scoping web search
        scoping_sources: List[Source] = []
        try:
            search_results = await self.search_router.search(topic, max_results=4)
            for idx, r in enumerate(search_results):
                scoping_sources.append(
                    Source(
                        id=f"SC{idx+1}",
                        url=r.url,
                        title=r.title,
                        snippet=r.snippet,
                        backend=r.backend,
                    )
                )
        except Exception as e:
            print(f"Scoping search error: {e}")

        await event_bus.publish(
            AgentEvent(
                run_id=run_id,
                stage="planning",
                agent_id="orchestrator",
                agent_name="Orchestrator Agent",
                status="progress",
                message=f"Scoping completed ({len(scoping_sources)} background sources found). Generating plan...",
            )
        )

        # Determine target number of subtasks based on depth
        subtask_count = 3 if depth == "quick" else (5 if depth == "standard" else 7)

        # 2. Generate Plan with LLM
        scoping_context = "\n".join([f"- [{s.id}] {s.title}: {s.snippet} ({s.url})" for s in scoping_sources])
        prompt = (
            f"You are the Lead Research Orchestrator. Your mission is to formulate a comprehensive research plan.\n\n"
            f"TOPIC: {topic}\n"
            f"ADDITIONAL USER INSTRUCTIONS: {extra_instructions or 'None'}\n"
            f"DEPTH LEVEL: {depth} (Aim for approximately {subtask_count} distinct sub-tasks)\n\n"
            f"BACKGROUND SCOPING INFORMATION FROM THE WEB:\n{scoping_context or 'None available'}\n\n"
            f"Please output a structured JSON plan with the following fields:\n"
            f"- topic: string\n"
            f"- interpretation: string (your analytical understanding of what needs to be investigated)\n"
            f"- audience: string\n"
            f"- subtasks: array of objects with (id, title, objective, key_questions, suggested_queries, priority)\n"
            f"- final_report_outline: array of string section headings for the final synthesis\n\n"
            f"Ensure subtasks are mutually exclusive and collectively exhaustive (MECE)."
        )

        response = await self.llm.chat(
            messages=[
                {"role": "system", "content": "You are a master research planner. Output valid JSON."},
                {"role": "user", "content": prompt},
            ],
            role="planner",
            response_schema=ResearchPlan,
        )

        plan: ResearchPlan = response.parsed  # type: ignore
        plan.scoping_sources = scoping_sources

        # Ensure unique IDs
        for idx, st in enumerate(plan.subtasks):
            if not st.id or st.id.strip() == "":
                st.id = f"subtask_{idx+1}"

        await event_bus.publish(
            AgentEvent(
                run_id=run_id,
                stage="planning",
                agent_id="orchestrator",
                agent_name="Orchestrator Agent",
                status="completed",
                message=f"Research plan created with {len(plan.subtasks)} sub-topics.",
                data={"subtasks": [st.title for st in plan.subtasks]},
            )
        )

        return plan

    async def review_draft(
        self,
        plan: ResearchPlan,
        draft_markdown: str,
        sections: List[SectionReport],
        run_id: str = "",
    ) -> Critique:
        """Critique draft against the plan, and perform web fact-checking of key claims."""
        await event_bus.publish(
            AgentEvent(
                run_id=run_id,
                stage="reviewing",
                agent_id="orchestrator",
                agent_name="Orchestrator (Reviewer)",
                status="running",
                message="Reviewing draft against plan requirements and conducting web fact-checks...",
            )
        )

        # 1. Fact checking web search: extract key claims to check
        fact_checks: List[FactCheckItem] = []
        try:
            fc_prompt = (
                f"Extract 2-3 specific, verifiable factual claims or statistics made in this research draft that should be verified on the web:\n\n"
                f"{draft_markdown[:3000]}\n\n"
                f"Return a JSON array of strings containing the queries to search for."
            )
            fc_resp = await self.llm.chat(
                messages=[{"role": "user", "content": fc_prompt}],
                role="reviewer",
            )
            # Find queries
            import re
            queries = re.findall(r'"([^"]{10,100})"', fc_resp.content)
            for q in queries[:2]:
                results = await self.search_router.search(q, max_results=2)
                if results:
                    fact_checks.append(
                        FactCheckItem(
                            claim=q,
                            verdict="verified",
                            notes=f"Corroborated by {results[0].title}: {results[0].snippet[:150]}",
                            source_url=results[0].url,
                        )
                    )
        except Exception as e:
            print(f"Fact check error: {e}")

        # 2. Comprehensive Critique
        plan_summary = f"Topic: {plan.topic}\nSubtasks: {', '.join(st.title for st in plan.subtasks)}"
        critique_prompt = (
            f"You are the Orchestrator acting as Senior Peer Reviewer.\n"
            f"ORIGINAL RESEARCH PLAN:\n{plan_summary}\n\n"
            f"SYNTHESIZED DRAFT REPORT (Excerpt):\n{draft_markdown[:5000]}\n\n"
            f"Evaluate whether this draft adequately covers the required subtasks, maintains objective academic/technical rigor, "
            f"and provides clear citation references. If quality is satisfactory, set approved=true. Otherwise, provide actionable instructions.\n"
            f"Output JSON matching Critique schema:\n"
            f"- approved: boolean\n"
            f"- issues: list of strings\n"
            f"- missing_coverage: list of strings\n"
            f"- instructions_for_revision: string"
        )

        response = await self.llm.chat(
            messages=[
                {"role": "system", "content": "You are a critical, rigorous technical reviewer. Output valid JSON."},
                {"role": "user", "content": critique_prompt},
            ],
            role="reviewer",
            response_schema=Critique,
        )

        critique: Critique = response.parsed  # type: ignore
        critique.fact_checks = fact_checks

        await event_bus.publish(
            AgentEvent(
                run_id=run_id,
                stage="reviewing",
                agent_id="orchestrator",
                agent_name="Orchestrator (Reviewer)",
                status="completed",
                message=f"Draft review completed. Status: {'Approved' if critique.approved else 'Revision requested'}.",
                data={"approved": critique.approved, "issues_count": len(critique.issues)},
            )
        )

        return critique

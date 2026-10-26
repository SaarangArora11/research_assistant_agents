"""Researcher Agent: Gathers sources, extracts notes, and generates comprehensive section reports."""

from __future__ import annotations
import asyncio
from typing import Dict, List, Optional
from app.core.events import event_bus
from app.llm.client import LLMClient
from app.models import AgentEvent, SectionReport, Source, SubTask
from app.tools.fetch.router import FetchRouter
from app.tools.search.router import SearchRouter


class ResearcherAgent:
    """Specialized subagent responsible for investigating a single SubTask."""

    def __init__(
        self,
        llm_client: LLMClient,
        search_router: SearchRouter,
        fetch_router: FetchRouter,
    ):
        self.llm = llm_client
        self.search_router = search_router
        self.fetch_router = fetch_router

    async def research(
        self,
        subtask: SubTask,
        overall_topic: str,
        max_sources: int = 5,
        run_id: str = "",
    ) -> SectionReport:
        agent_id = f"researcher_{subtask.id}"
        agent_name = f"Researcher: {subtask.title[:25]}"

        await event_bus.publish(
            AgentEvent(
                run_id=run_id,
                stage="researching",
                agent_id=agent_id,
                agent_name=agent_name,
                status="started",
                message=f"Starting research on '{subtask.title}'",
            )
        )

        # 1. Gather search results
        queries = list(subtask.suggested_queries)
        if not queries:
            queries = [f"{overall_topic} {subtask.title}"]

        discovered_sources: List[Source] = []
        seen_urls = set()

        for q in queries[:3]:
            await event_bus.publish(
                AgentEvent(
                    run_id=run_id,
                    stage="researching",
                    agent_id=agent_id,
                    agent_name=agent_name,
                    status="progress",
                    message=f"Searching web: '{q}'",
                )
            )
            try:
                results = await self.search_router.search(q, max_results=3)
                for r in results:
                    if r.url not in seen_urls:
                        seen_urls.add(r.url)
                        source_id = f"S{len(discovered_sources) + 1}"
                        discovered_sources.append(
                            Source(
                                id=source_id,
                                url=r.url,
                                title=r.title,
                                snippet=r.snippet,
                                backend=r.backend,
                            )
                        )
                        if len(discovered_sources) >= max_sources:
                            break
            except Exception as e:
                print(f"Researcher search error for '{q}': {e}")

            if len(discovered_sources) >= max_sources:
                break

        # 2. Fetch page contents in parallel
        await event_bus.publish(
            AgentEvent(
                run_id=run_id,
                stage="researching",
                agent_id=agent_id,
                agent_name=agent_name,
                status="progress",
                message=f"Reading full content for {len(discovered_sources)} web sources...",
            )
        )

        fetch_tasks = [self.fetch_router.fetch_url(s.url) for s in discovered_sources]
        fetched_texts = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        source_notes: List[str] = []
        for idx, (source, content) in enumerate(zip(discovered_sources, fetched_texts)):
            if isinstance(content, str) and content.strip():
                # Extract summary / key points for this source
                source_notes.append(
                    f"--- Source [{source.id}]: {source.title} ({source.url}) ---\n"
                    f"Excerpt: {content[:3500]}\n"
                )
            else:
                # Fallback to snippet
                source_notes.append(
                    f"--- Source [{source.id}]: {source.title} ({source.url}) ---\n"
                    f"Snippet: {source.snippet}\n"
                )

        combined_notes = "\n".join(source_notes)

        # 3. Generate Section Report
        await event_bus.publish(
            AgentEvent(
                run_id=run_id,
                stage="researching",
                agent_id=agent_id,
                agent_name=agent_name,
                status="progress",
                message=f"Drafting section report with citations for '{subtask.title}'...",
            )
        )

        questions_text = "\n".join([f"- {q}" for q in subtask.key_questions])
        prompt = (
            f"You are an expert scientific/technical researcher specializing in: '{subtask.title}'.\n\n"
            f"OVERALL TOPIC: {overall_topic}\n"
            f"SUB-TOPIC OBJECTIVE: {subtask.objective}\n\n"
            f"KEY QUESTIONS TO ANSWER:\n{questions_text}\n\n"
            f"RESEARCH EVIDENCE GATHERED FROM THE WEB:\n"
            f"{combined_notes}\n\n"
            f"Please write a comprehensive, rigorous section report in Markdown.\n"
            f"CRITICAL RULES:\n"
            f"1. You MUST explicitly cite sources in the text using bracketed format [S1], [S2], etc., corresponding to the sources above.\n"
            f"2. Present deep factual insights, numbers, comparative data, and mechanisms where available.\n"
            f"3. Highlight limitations, conflicting findings, or open questions.\n\n"
            f"Return a JSON object conforming to SectionReport schema:\n"
            f"- subtask_id: '{subtask.id}'\n"
            f"- title: '{subtask.title}'\n"
            f"- markdown: string (full section text with [S#] citations)\n"
            f"- key_findings: list of bullet strings\n"
            f"- open_questions: list of strings\n"
            f"- confidence: 'low', 'medium', or 'high'"
        )

        response = await self.llm.chat(
            messages=[
                {"role": "system", "content": "You are a meticulous investigative researcher. Output valid JSON."},
                {"role": "user", "content": prompt},
            ],
            role="researcher",
            response_schema=SectionReport,
        )

        report: SectionReport = response.parsed  # type: ignore
        report.sources = discovered_sources
        report.subtask_id = subtask.id
        report.title = subtask.title

        await event_bus.publish(
            AgentEvent(
                run_id=run_id,
                stage="researching",
                agent_id=agent_id,
                agent_name=agent_name,
                status="completed",
                message=f"Section '{subtask.title}' completed with {len(report.sources)} cited sources.",
                data={"confidence": report.confidence, "sources_count": len(report.sources)},
            )
        )

        return report

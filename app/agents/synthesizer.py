"""Synthesizer Agent: Combines section reports into a unified, coherent final research report."""

from __future__ import annotations
import re
from typing import Dict, List, Optional, Tuple
from app.core.events import event_bus
from app.llm.client import LLMClient
from app.models import AgentEvent, Critique, ResearchPlan, SectionReport, Source


class SynthesizerAgent:
    """Agent that merges, deduplicates, and weaves individual section reports into a final document."""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def _unify_citations(
        self, sections: List[SectionReport]
    ) -> Tuple[List[SectionReport], List[Source]]:
        """Map all section sources to a unified global list [S1...SN] and rewrite markdown."""
        global_sources: List[Source] = []
        url_to_global_id: Dict[str, str] = {}
        updated_sections: List[SectionReport] = []

        for sec in sections:
            # Local id to global id mapping for this section
            local_to_global: Dict[str, str] = {}

            for src in sec.sources:
                clean_url = src.url.strip()
                if clean_url not in url_to_global_id:
                    new_id = f"S{len(global_sources) + 1}"
                    url_to_global_id[clean_url] = new_id
                    new_src = src.model_copy()
                    new_src.id = new_id
                    global_sources.append(new_src)
                local_to_global[src.id] = url_to_global_id[clean_url]

            # Rewrite citations in markdown
            rewritten_md = sec.markdown
            # Sort local keys by length descending to prevent partial prefix replacements
            for local_id in sorted(local_to_global.keys(), key=len, reverse=True):
                global_id = local_to_global[local_id]
                pattern = rf"\[{re.escape(local_id)}\]"
                rewritten_md = re.sub(pattern, f"[{global_id}]", rewritten_md)

            sec_copy = sec.model_copy()
            sec_copy.markdown = rewritten_md
            updated_sections.append(sec_copy)

        return updated_sections, global_sources

    async def synthesize(
        self,
        plan: ResearchPlan,
        sections: List[SectionReport],
        previous_draft: Optional[str] = None,
        critique: Optional[Critique] = None,
        user_feedback: Optional[str] = None,
        run_id: str = "",
    ) -> str:
        """Synthesize section reports into a polished, unified markdown report."""
        await event_bus.publish(
            AgentEvent(
                run_id=run_id,
                stage="synthesizing",
                agent_id="synthesizer",
                agent_name="Synthesizer Agent",
                status="started",
                message="Harmonizing citations and synthesizing section reports...",
            )
        )

        unified_sections, global_sources = self._unify_citations(sections)

        # Build sections digest
        sections_digest = []
        for s in unified_sections:
            findings_str = "\n".join([f"  * {f}" for f in s.key_findings])
            sections_digest.append(
                f"### SUB-TOPIC: {s.title} (Confidence: {s.confidence})\n"
                f"Key Findings:\n{findings_str}\n\n"
                f"Section Body:\n{s.markdown}\n"
            )
        all_sections_text = "\n\n".join(sections_digest)

        # Build references block
        references_lines = ["\n\n## References\n"]
        for src in global_sources:
            title = src.title or "Web Source"
            references_lines.append(f"- **[{src.id}]** [{title}]({src.url}) — {src.snippet[:120]}...")
        references_block = "\n".join(references_lines)

        # Revision context if applicable
        revision_instruction = ""
        if critique and not critique.approved:
            issues = "\n".join([f"- {iss}" for iss in critique.issues])
            missing = "\n".join([f"- {m}" for m in critique.missing_coverage])
            revision_instruction = (
                f"\n\nCRITIQUE FEEDBACK FROM PREVIOUS REVIEW:\n"
                f"Identified Issues:\n{issues}\n"
                f"Missing Coverage:\n{missing}\n"
                f"Instructions: {critique.instructions_for_revision}\n"
            )

        if user_feedback:
            revision_instruction += f"\n\nDIRECT USER REVISION FEEDBACK:\n{user_feedback}\n"

        prompt = (
            f"You are the Chief Research Synthesizer. Your task is to merge the specialized section reports "
            f"into a single, high-impact, cohesive research report in Markdown.\n\n"
            f"TOPIC: {plan.topic}\n"
            f"AUDIENCE: {plan.audience}\n"
            f"RECOMMENDED REPORT OUTLINE:\n{chr(10).join(['- ' + o for o in plan.final_report_outline])}\n"
            f"{revision_instruction}\n"
            f"SECTION REPORTS WITH UNIFIED CITATIONS:\n"
            f"{all_sections_text}\n\n"
            f"WRITING REQUIREMENTS:\n"
            f"1. Create an engaging `# {plan.topic}` title followed by an Executive Summary.\n"
            f"2. Synthesize themes together logically rather than just pasting disconnected sections.\n"
            f"3. Eliminate redundancies while preserving deep technical details and factual accuracy.\n"
            f"4. You MUST retain the bracketed citations `[S1]`, `[S2]`, etc. exactly where evidence is cited.\n"
            f"5. End with a Strategic Implications / Key Takeaways section.\n"
            f"6. Do NOT include the references section yet; it will be appended automatically.\n\n"
            f"Write the complete, comprehensive markdown report now:"
        )

        response = await self.llm.chat(
            messages=[
                {"role": "system", "content": "You are a world-class scientific and technical writer and synthesizer."},
                {"role": "user", "content": prompt},
            ],
            role="synthesizer",
            max_tokens=6000,
        )

        final_body = response.content.strip()
        # Append references block
        complete_report = f"{final_body}{references_block}"

        await event_bus.publish(
            AgentEvent(
                run_id=run_id,
                stage="synthesizing",
                agent_id="synthesizer",
                agent_name="Synthesizer Agent",
                status="completed",
                message=f"Report synthesized successfully ({len(global_sources)} total cited sources).",
                data={"sources_count": len(global_sources), "word_count": len(complete_report.split())},
            )
        )

        return complete_report

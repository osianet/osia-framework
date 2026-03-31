"""
Source tracking and quality auditing for OSIA intelligence reports.

Captures provenance metadata during the research phase and provides
post-processing validation of citations in final reports.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum

logger = logging.getLogger("osia.sources")


class SourceReliability(str, Enum):  # noqa: UP042 — StrEnum requires Python 3.11+, keeping str+Enum for compatibility
    """Admiralty-style source reliability rating."""

    A = "A — Peer-reviewed / Official government source"
    B = "B — Established media / Institutional publication"
    C = "C — Web search / Blog / Unverified news outlet"
    D = "D — Social media / User-generated content"
    E = "E — Unverifiable / Single-source claim"


# Map tool names to default reliability tiers
TOOL_RELIABILITY: dict[str, SourceReliability] = {
    "search_arxiv": SourceReliability.A,
    "search_semantic_scholar": SourceReliability.A,
    "search_wikipedia": SourceReliability.B,
    "search_web": SourceReliability.C,
    "get_youtube_transcript": SourceReliability.C,
    "read_social_comments": SourceReliability.D,
    "read_social_post": SourceReliability.D,
    "post_social_comment": SourceReliability.D,
    "reply_social_comment": SourceReliability.D,
}


@dataclass
class SourceRecord:
    """A single source captured during research."""

    tool: str
    query: str
    snippet: str
    reliability: SourceReliability
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def format_short(self, index: int) -> str:
        return f'[{index}] ({self.reliability.name}) {self.tool} — q: "{self.query}" | {self.snippet[:120]}'


class SourceTracker:
    """Accumulates source records during a research loop."""

    def __init__(self):
        self.sources: list[SourceRecord] = []

    def record(self, tool_name: str, query: str, result_text: str):
        """Log a tool call and its result as a source."""
        reliability = TOOL_RELIABILITY.get(tool_name, SourceReliability.E)
        # First meaningful line as snippet (skip blanks)
        snippet = ""
        for line in result_text.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("Tool error"):
                snippet = stripped
                break
        if not snippet:
            snippet = result_text[:150].strip()

        record = SourceRecord(
            tool=tool_name,
            query=query,
            snippet=snippet,
            reliability=reliability,
        )
        self.sources.append(record)
        logger.debug("Recorded source [%s] %s — %s", reliability.name, tool_name, query)

    def format_manifest(self) -> str:
        """Render the full source manifest for injection into desk prompts."""
        if not self.sources:
            return ""
        lines = ["## Research Source Manifest", ""]
        for i, src in enumerate(self.sources, 1):
            lines.append(src.format_short(i))
        lines.append("")
        lines.append(
            f"Total sources: {len(self.sources)} | "
            f"A-tier: {self._count_tier('A')} | "
            f"B-tier: {self._count_tier('B')} | "
            f"C-tier: {self._count_tier('C')} | "
            f"D-tier: {self._count_tier('D')}"
        )
        return "\n".join(lines)

    def _count_tier(self, tier: str) -> int:
        return sum(1 for s in self.sources if s.reliability.name == tier)


def build_citation_protocol() -> str:
    """Return citation guidance appended to every desk prompt."""
    return (
        "\n\n--- CITATION GUIDANCE ---\n"
        "Attribute sources where provenance meaningfully affects confidence in the assessment. "
        "Use bracketed numbers [1], [2] for specific, traceable claims and list them in a '## Sources' "
        "section when your report has enough distinct sources to warrant one. "
        "Reliability scale: A — Peer-reviewed/Official | B — Established media | C — Web/Blog | "
        "D — Social media | E — Unverifiable. Format: [N] (Rating) Origin — Description.\n"
        "Add '## Source Confidence' (HIGH / MODERATE / LOW) only when the overall quality of your source "
        "pool is meaningful to the reader. For shorter reports or when synthesising from provided "
        "intelligence context, inline attribution is fine — do not force a formal citation block where "
        "it adds no value. Mark genuinely unsourced claims as [UNSOURCED] where that gap matters. "
        "Never fabricate citations.\n"
    )


def audit_report(report_text: str | None, tracker: SourceTracker | None = None) -> str:
    """Append a source summary to a finished report, only when meaningful.

    Only appends when research sources were actually gathered. Logs a debug
    warning if citations appear absent despite sources being available.
    """
    if not report_text:
        return ""

    # Only append anything if research was actually performed
    if not tracker or not tracker.sources:
        return ""

    citation_refs = set(re.findall(r"\[(\d+)\]", report_text))
    manifest_count = len(tracker.sources)
    cited_count = len(citation_refs)

    # Log a debug note if sources were gathered but none cited — don't surface to output
    if cited_count == 0:
        logger.debug("audit_report: %d sources gathered but no citations in report", manifest_count)

    tier_summary = (
        f"Research pool: {tracker._count_tier('A')}×A, "
        f"{tracker._count_tier('B')}×B, "
        f"{tracker._count_tier('C')}×C, "
        f"{tracker._count_tier('D')}×D"
    )
    lines = [f"\n\n*Source pool: {manifest_count} references ({tier_summary})*"]

    if cited_count < manifest_count and cited_count > 0:
        lines.append(f"*{manifest_count - cited_count} gathered sources not cited in report.*")

    return "\n".join(lines)

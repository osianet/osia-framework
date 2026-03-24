"""
Source tracking and quality auditing for OSIA intelligence reports.

Captures provenance metadata during the research phase and provides
post-processing validation of citations in final reports.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger("osia.sources")


class SourceReliability(str, Enum):
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
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def format_short(self, index: int) -> str:
        return f"[{index}] ({self.reliability.name}) {self.tool} — q: \"{self.query}\" | {self.snippet[:120]}"


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
        lines.append(f"Total sources: {len(self.sources)} | "
                      f"A-tier: {self._count_tier('A')} | "
                      f"B-tier: {self._count_tier('B')} | "
                      f"C-tier: {self._count_tier('C')} | "
                      f"D-tier: {self._count_tier('D')}")
        return "\n".join(lines)

    def _count_tier(self, tier: str) -> int:
        return sum(1 for s in self.sources if s.reliability.name == tier)


def build_citation_protocol() -> str:
    """Return the standardized citation instructions appended to every desk prompt."""
    return (
        "\n\n--- CITATION PROTOCOL ---\n"
        "You MUST follow this citation protocol in every report:\n"
        "1. Tag each factual claim with a bracketed citation number, e.g. [1], [2].\n"
        "2. At the end of your report, include a '## Sources' section listing every cited source.\n"
        "3. For each source entry, include:\n"
        "   - The citation number\n"
        "   - The origin (tool name, URL, or RSS feed)\n"
        "   - A reliability rating using this scale:\n"
        "     A — Peer-reviewed / Official government source\n"
        "     B — Established media / Institutional publication\n"
        "     C — Web search / Blog / Unverified news outlet\n"
        "     D — Social media / User-generated content\n"
        "     E — Unverifiable / Single-source claim\n"
        "   Format: [N] (Rating) Origin — Description\n"
        "4. After the Sources section, add a one-line '## Source Confidence' assessment:\n"
        "   HIGH (mostly A/B sources), MODERATE (mixed), or LOW (mostly C/D/E).\n"
        "5. If a claim cannot be sourced, mark it as [UNSOURCED] inline.\n"
        "6. Never fabricate citations. If you lack a source, say so.\n"
    )


def audit_report(report_text: str, tracker: SourceTracker | None = None) -> str:
    """Append a source audit summary to a finished report.

    Checks for citation markers in the report body and cross-references
    against the research source manifest when available.
    """
    # Count inline citations like [1], [2], etc.
    citation_refs = set(re.findall(r"\[(\d+)\]", report_text))
    has_sources_section = bool(re.search(r"##\s*Sources", report_text, re.IGNORECASE))
    has_confidence = bool(re.search(r"##\s*Source Confidence", report_text, re.IGNORECASE))
    unsourced_count = len(re.findall(r"\[UNSOURCED\]", report_text, re.IGNORECASE))

    audit_lines = ["\n\n--- SOURCE AUDIT ---"]

    if citation_refs:
        audit_lines.append(f"Citations found: {len(citation_refs)} unique references")
    else:
        audit_lines.append("⚠️ NO CITATIONS DETECTED in report body")

    if has_sources_section:
        audit_lines.append("Sources section: PRESENT")
    else:
        audit_lines.append("⚠️ Sources section: MISSING")

    if has_confidence:
        audit_lines.append("Source confidence rating: PRESENT")
    else:
        audit_lines.append("⚠️ Source confidence rating: MISSING")

    if unsourced_count:
        audit_lines.append(f"Unsourced claims flagged: {unsourced_count}")

    # Cross-reference with research manifest if available
    if tracker and tracker.sources:
        manifest_count = len(tracker.sources)
        cited_count = len(citation_refs)
        if cited_count < manifest_count:
            audit_lines.append(
                f"Note: {manifest_count} sources were gathered during research, "
                f"but only {cited_count} citations appear in the report."
            )
        tier_summary = (
            f"Research pool: {tracker._count_tier('A')}×A, "
            f"{tracker._count_tier('B')}×B, "
            f"{tracker._count_tier('C')}×C, "
            f"{tracker._count_tier('D')}×D"
        )
        audit_lines.append(tier_summary)

    return "\n".join(audit_lines)

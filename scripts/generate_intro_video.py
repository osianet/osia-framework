"""
Generate OSIA intro webcast video.

Produces a structured briefing with:
1. Host intro (what is OSIA, how it works)
2. Each desk head introduces themselves with portrait (45 sec each)
3. Demo of capabilities (intel flow, entity extraction, Hermes validation)
4. Closing remarks

Follows the existing briefing_generator pattern: generates narration via Chatterbox TTS,
renders slides with portraits, assembles video via ffmpeg.

Usage:
    uv run python scripts/generate_intro_video.py --dry-run
    uv run python scripts/generate_intro_video.py
"""

import argparse
import json
import logging
import re
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("osia.intro_video")

ROOT = Path(__file__).parent.parent
CONFIG_DIR = ROOT / "config" / "desks"
OUT_DIR = ROOT / "reports" / "intro_webcast"


def load_desk_configs() -> dict[str, dict]:
    """Load all desk configs."""
    desks = {}
    for yaml_file in sorted(CONFIG_DIR.glob("*.yaml")):
        with open(yaml_file) as f:
            cfg = yaml.safe_load(f)
            if cfg and cfg.get("briefing"):
                desks[cfg["slug"]] = cfg
    return desks


def _extract_name(persona: str) -> str:
    """Extract full name from persona text."""
    match = re.search(
        r"(?:Director|Dr\.|Commander|Agent|Professor)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        persona,
    )
    return match.group(0) if match else "Department Head"


def _desk_focus(slug: str) -> str:
    """Return a one-line focus for each desk."""
    focuses = {
        "geopolitical-and-security-desk": "statecraft, military operations, and international relations",
        "cyber-intelligence-and-warfare-desk": "nation-state cyber operations and digital threats",
        "human-intelligence-and-profiling-desk": "behavioral profiling and power networks",
        "finance-and-economics-directorate": "financial flows, sanctions, and economic warfare",
        "environment-and-ecology-desk": "environmental threats and ecological intelligence",
        "cultural-and-theological-intelligence-desk": "cultural movements and ideological drivers",
        "science-technology-and-commercial-desk": "technical breakthroughs and dual-use technology",
        "information-warfare-desk": "propaganda, influence operations, and narrative warfare",
        "the-watch-floor": "synthesis and strategic assessment",
    }
    return focuses.get(slug, "intelligence analysis")


def _desk_narration(slug: str, name: str, desk_name: str) -> str:
    """Return a persona-voiced narration for each desk head's introduction."""
    narrations = {
        "geopolitical-and-security-desk": f"""
{name}. Geopolitical and Security Desk.

Thirty years tracking how states behave when they think no one is watching. The declarations
matter less than the deployments. The communiqués matter less than the troop movements.

My desk covers state-on-state conflict, alliance realignment, military doctrine, and the
spaces where formal diplomacy breaks down and covert pressure begins. When an actor's public
position and their actual behaviour diverge — that's where we work.

Open sources tell you more than most people expect, if you know how to read them.
""",
        "cyber-intelligence-and-warfare-desk": f"""
Commander {name.split()[-1]}, Cyber Intelligence and Warfare.

I don't lead with the malware. I lead with who ran it, why, and what it tells us about
their next move. The technical detail matters — but only as evidence of intent.

We track nation-state actors, their tooling, their infrastructure, and the strategic
objectives behind their operations. A zero-day is just a weapon. The question is who
bought it, and what they're pointing it at.

If there's a significant cyber incident anywhere in the world, we'll have an assessment
on the Watch Floor within hours.
""",
        "human-intelligence-and-profiling-desk": f"""
{name}. Human Intelligence and Profiling.

People are never as opaque as they think they are. Power structures leave traces —
in relationships, in contradictions, in the gaps between what someone says publicly
and what they do under pressure.

My desk builds profiles: who holds real influence, who's being leveraged, who's about
to flip. We map networks, track behavioral shifts, and identify the pressure points
that matter before an event, not after.

The most useful intelligence is rarely the dramatic kind. It's the quiet pattern you
noticed six months before anyone else was paying attention.
""",
        "finance-and-economics-directorate": f"""
{name}, Finance and Economics Directorate.

Follow the money. That's the entire methodology.

Shell companies, correspondent banking relationships, sanctions evasion routes,
dark money flows into political campaigns — the financial system is more transparent
than most actors believe, if you're willing to do the work.

My desk connects capital movements to the political interests they serve. Economic
policy is foreign policy. Debt is leverage. We track who controls what, who owes
what to whom, and where the pressure will land when it's applied.

The surface explanation is rarely the right one. We keep going.
""",
        "environment-and-ecology-desk": f"""
{name}, Environment and Ecological Intelligence.

The environment is an early warning system. By the time something becomes a
humanitarian crisis, the ecological signals were there months or years before.

Aquifer depletion, crop failures, deforestation rates, coastal erosion — I read
these as intelligence, not just data. They predict displacement, resource conflict,
infrastructure failure, and political instability with more lead time than most
traditional indicators.

My desk translates what the planet is telling us into strategic assessments. The
ecological crisis is already a security crisis. We're mapping what that looks like
on the ground, now and over the next decade.
""",
        "cultural-and-theological-intelligence-desk": f"""
{name}, Cultural and Theological Intelligence.

Culture and religion are treated as background noise in most intelligence frameworks.
That's a significant error.

Movements don't emerge from nowhere. They have deep historical roots, theological
frameworks, and cultural logic that shapes how they organise, how they recruit, and
how they justify action. If you don't understand that logic on its own terms, you
will consistently misread what you're seeing.

I bring both academic rigour and field experience to this desk. We contextualise
what's happening within its actual history and spiritual frameworks — not the
simplified version that fits a preexisting narrative.

The ideological dimension of any conflict is as operationally relevant as the military one.
""",
        "science-technology-and-commercial-desk": f"""
{name}, Science, Technology and Commercial Desk.

The most consequential shifts in the global balance of power right now are technical.
AI, synthetic biology, quantum computing, satellite infrastructure — these aren't
peripheral concerns. They're the terrain.

My job is translation. I take what's happening in laboratories, patent filings,
academic preprints, and commercial deals, and I tell you what it means strategically.
Who controls this technology? Who does it empower? Who does it leave behind, or
actively threaten?

Dual-use technology is the grey zone we live in. Understanding it requires staying
genuinely current — and being willing to follow the implications wherever they lead.
""",
        "information-warfare-desk": f"""
{name}. Information and Psychological Warfare.

The information environment is a battlefield. Most of the casualties don't know
they've been hit.

My desk tracks propaganda campaigns, coordinated inauthentic behaviour, narrative
laundering, and influence operations — from state broadcasters to anonymous Telegram
channels. We map the infrastructure: who's running the accounts, who's funding
the outlets, and whose interests the messaging serves.

There's a certain irony in doing this work with open sources. Everything I need
to understand an influence operation is usually visible if you know what patterns
to look for. The operators count on the noise to provide cover.

We cut through the noise.
""",
        "the-watch-floor": f"""
{name}. The Watch Floor.

I see everything.

Every desk report, every entity flag, every corroboration verdict from Hermes —
it all comes through here. My job is synthesis: taking nine different pictures
of a complex event and producing one coherent assessment that tells decision-makers
what is actually happening and what matters.

No single desk has the full picture. Geopolitical context without the financial
layer misses the leverage. Cyber attribution without the human network misses the
command structure. The Watch Floor holds all of it simultaneously.

When the agency delivers a finished intelligence product, this is where it was
assembled. I speak last, because I speak for all of them.
""",
    }
    return narrations.get(
        slug,
        f"""
{name}, {desk_name}.

My desk covers {_desk_focus(slug)}. We work across open sources, monitoring for
developments that matter to the agency's intelligence picture.

Every finding we produce goes through corroboration before it reaches the Watch Floor.
""",
    )


_HOST_VOICE_REF = str(ROOT / "config" / "voice_refs" / "intro-host.mp3")


def build_slides() -> list[dict]:
    """Build slide deck for intro video."""
    desks = load_desk_configs()
    slides = []

    # Slide 1: Title
    slides.append(
        {
            "slide_type": "title",
            "title": "Open Source Intelligence Agency",
            "body": "An Autonomous Intelligence Framework",
            "bg_category": "hero",
            "voice_ref_path": _HOST_VOICE_REF,
            "narration": """
Welcome to the Open Source Intelligence Agency — OSIA.

We are an autonomous, event-driven intelligence framework that automates the collection,
analysis, and reporting of open-source intelligence. Think of us as a real intelligence
agency, but built entirely on open data and AI.

Let me walk you through how we work.
""",
        }
    )

    # Slide 2: How it works
    slides.append(
        {
            "slide_type": "content",
            "title": "The Intelligence Lifecycle",
            "bg_category": "data_overlay",
            "voice_ref_path": _HOST_VOICE_REF,
            "body": """
- **Ingress**: Signal, RSS feeds, or API
- **Research**: Multi-turn loops via MCP tools
- **Entity Extraction**: Automated discovery and research queueing
- **Validation**: Hermes worker corroborates findings
- **Analysis**: Specialized desks synthesize intelligence
- **Synthesis**: Watch Floor creates unified INTSUM
- **Delivery**: Reports archived and delivered
""",
            "narration": """
Intelligence enters through three channels: Signal messenger, RSS feeds, and our API.

Each request flows through our Chief of Staff — an orchestrator that routes tasks to
specialized AI desks. Each desk is an expert in their domain: geopolitics, cyber threats,
finance, human networks, culture, science, ecology, and information warfare.

When a desk receives a task, it launches a research loop. Our agents search Wikipedia,
ArXiv, Tavily, and other open sources. We extract entities — people, organizations,
locations — and enqueue them for deeper research.

Our Hermes worker then validates findings against multiple sources, upgrading confidence
tiers as evidence accumulates. Every finding is embedded and stored in Qdrant, our vector
database, with temporal decay scoring so recent intelligence ranks above stale entries.

Finally, our Watch Floor synthesizes all desk reports into a unified intelligence summary.
The finished analysis is archived and delivered back to the requester.

Now let me introduce you to the team.
""",
        }
    )

    # Slides 3-11: Each desk head
    desk_order = [
        "geopolitical-and-security-desk",
        "cyber-intelligence-and-warfare-desk",
        "human-intelligence-and-profiling-desk",
        "finance-and-economics-directorate",
        "environment-and-ecology-desk",
        "cultural-and-theological-intelligence-desk",
        "science-technology-and-commercial-desk",
        "information-warfare-desk",
        "the-watch-floor",
    ]

    for slug in desk_order:
        if slug not in desks:
            continue
        desk = desks[slug]
        briefing = desk.get("briefing", {})
        persona = briefing.get("persona", "").strip()
        name = _extract_name(persona)

        slides.append(
            {
                "slide_type": "content",
                "title": desk["name"],
                "body": f"**{name}**\n\n{_desk_focus(slug)}",
                "narration": _desk_narration(slug, name, desk["name"]),
                "desk_slug": slug,
                "desk_name": desk["name"],
                "persona_name": name,
            }
        )

    # Slide 12: Capabilities demo
    slides.append(
        {
            "slide_type": "content",
            "bg_category": "archive",
            "voice_ref_path": _HOST_VOICE_REF,
            "title": "How We Validate Intelligence",
            "body": """
- **Phase 1**: Search internal KB + external sources
- **Phase 2**: Structured verdict from Hermes 4
- **Verdicts**: CORROBORATED, CONTRADICTED, UNVERIFIED
- **Confidence Tiers**: A (high), B (moderate), C (low)
- **Temporal Decay**: Recent intel ranks above stale entries
""",
            "narration": """
Our Hermes worker runs a two-phase corroboration loop on every finding.

Phase 1 searches our internal knowledge base plus external sources like Tavily, OTX, ArXiv,
and Aleph. Phase 2 issues a structured verdict: is this finding corroborated by independent
sources, contradicted by evidence, or unverified?

We upgrade confidence tiers as evidence accumulates. A finding starts at tier B — moderate
confidence. If corroborated, it moves to tier A. If contradicted, it drops to tier C and
gets flagged for review.

This continuous validation ensures our intelligence picture stays accurate and current.
""",
        }
    )

    # Slide 13: Knowledge bases
    slides.append(
        {
            "slide_type": "content",
            "bg_category": "archive",
            "voice_ref_path": _HOST_VOICE_REF,
            "title": "Our Knowledge Bases",
            "body": """
- WikiLeaks cables (124K diplomatic records)
- Epstein files (declassified documents)
- MITRE ATT&CK (1.4K cyber techniques)
- CVE database (280K vulnerabilities)
- HackerOne reports (12.6K disclosures)
- Cybersecurity incidents (13.4K attacks)
- And growing...
""",
            "narration": """
We maintain specialized knowledge bases across multiple domains.

WikiLeaks cables give us historical diplomatic context. The Epstein files provide declassified
government documents. MITRE ATT&CK maps cyber techniques and threat actors. Our CVE database
tracks vulnerabilities. HackerOne reports show real-world exploits. And we're continuously
adding new sources.

Every desk can query these knowledge bases for context, ensuring our analysis is grounded
in evidence.
""",
        }
    )

    # Slide 14: Closing
    slides.append(
        {
            "slide_type": "closing",
            "bg_category": "ecological",
            "voice_ref_path": _HOST_VOICE_REF,
            "title": "The Future of OSIA",
            "body": """
Continuously expanding. Continuously learning. Continuously improving.

Intelligence for the people, the land, and the future.
""",
            "narration": """
OSIA is an ongoing project. We are continuously expanding our knowledge bases, adding new
desks, improving our validation pipeline, and scaling our infrastructure.

We operate under a decolonial and socialist mandate. Our analysis prioritizes anti-imperialism,
labor rights, ecological justice, and data sovereignty. We expose oppression, not suppress it.

If you're interested in contributing to OSIA — whether as a developer, analyst, or partner —
visit our GitHub repository to learn more.

Thank you for watching. This is OSIA: Intelligence for the people, the land, and the future.
""",
        }
    )

    return slides


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate OSIA intro webcast")
    parser.add_argument("--dry-run", action="store_true", help="Print slides without generating video")
    parser.add_argument("--generate", action="store_true", help="Generate full video with TTS + ffmpeg")
    parser.add_argument("--orientation", choices=["landscape", "portrait"], default="landscape")
    parser.add_argument("--resume", action="store_true", help="Skip existing files")
    args = parser.parse_args()

    slides = build_slides()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save slides
    slides_path = OUT_DIR / "intro_slides.json"
    with open(slides_path, "w") as f:
        json.dump(slides, f, indent=2)
    logger.info("Slides saved: %s", slides_path)

    if args.dry_run:
        print("\n=== OSIA INTRO WEBCAST SLIDES ===\n")
        for i, slide in enumerate(slides, 1):
            print(f"\n[SLIDE {i}] {slide['title']}")
            print(f"Type: {slide['slide_type']}\n")
            print(slide["narration"].strip())
            print("\n" + "=" * 80)
    elif args.generate:
        import asyncio
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from src.intelligence.intro_video_generator import generate_intro_video

        video_path = asyncio.run(generate_intro_video(slides, orientation=args.orientation, resume=args.resume))
        if video_path:
            print(f"\n✓ Video generated: {video_path}")
        else:
            print("\n✗ Video generation failed")
    else:
        logger.info("Slides generated. Use --generate to create video with TTS + ffmpeg.")


if __name__ == "__main__":
    main()

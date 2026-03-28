"""
OSIA Cyber Desk Seed Script

Seeds the cyber-intelligence-and-warfare-desk with background research by
submitting curated topics to the ingress API's /research and /ingest endpoints.

Research topics → osia:research_queue → research worker → osia_research_cache
  (builds background knowledge; model: mistral-31-24b via Venice)

Ingest queries → osia:task_queue → orchestrator → cyber desk → Signal/Qdrant
  (produces immediate INTSUMs; model: mistral-31-24b primary)

Cross-collection retrieval is already enabled on the cyber desk YAML, so all
existing collections (mitre-attack, cybersecurity-attacks, hackerone-reports,
cti-reports, ttp-mappings, cve-database, cti-bench) are automatically searched
at query time — these seeds add synthesised intelligence on top.

Usage:
  uv run python scripts/seed_cyber_desk.py                   # research mode (default)
  uv run python scripts/seed_cyber_desk.py --mode ingest     # ingest mode (INTSUMs)
  uv run python scripts/seed_cyber_desk.py --mode both       # research + ingest
  uv run python scripts/seed_cyber_desk.py --dry-run         # print topics, no requests
  uv run python scripts/seed_cyber_desk.py --delay 5         # seconds between requests (min 3s — 20/min rate limit)
  uv run python scripts/seed_cyber_desk.py --category apt    # specific category only
  uv run python scripts/seed_cyber_desk.py --list-categories # show available categories
"""

import argparse
import logging
import os
import sys
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("osia.seed_cyber")

INGRESS_API_URL = os.getenv("INGRESS_API_URL", "http://localhost:8097")
INGRESS_API_TOKEN = os.getenv("INGRESS_API_TOKEN", "")
INGRESS_API_UA = os.getenv("INGRESS_API_UA_SENTINEL", "osia-ingress/1")
CYBER_DESK = "cyber-intelligence-and-warfare-desk"

# ---------------------------------------------------------------------------
# Seed topics — grouped by intelligence category.
#
# research: background knowledge → osia_research_cache (safe to run in bulk)
# ingest:   immediate INTSUM → Signal + cyber desk Qdrant collection (use sparingly)
# ---------------------------------------------------------------------------

RESEARCH_TOPICS: dict[str, list[str]] = {
    "apt": [
        "Volt Typhoon TTPs and pre-positioning in US critical infrastructure 2023-2025",
        "Lazarus Group cryptocurrency theft operations and North Korean IT worker networks",
        "Sandworm destructive attacks against Ukrainian energy infrastructure",
        "APT28 (Fancy Bear) spearphishing and credential harvesting campaigns 2024-2025",
        "APT41 dual espionage and financial crime operations — state nexus analysis",
        "Salt Typhoon telecom sector intrusions and lawful intercept system compromise",
        "Scattered Spider social engineering and SIM swapping against enterprise targets",
        "APT29 (Cozy Bear) cloud infrastructure abuse and MFA bypass techniques",
        "Kimsuky intelligence collection operations against South Korean and US targets",
        "UNC3886 VMware ESXi zero-day exploitation and firmware persistence",
    ],
    "ttps": [
        "Living-off-the-land (LOTL) techniques: LOLBins, signed binary proxy execution",
        "Supply chain compromise patterns: SolarWinds, XZ Utils, 3CX — attack chain analysis",
        "Kerberoasting and AS-REP roasting Active Directory credential attacks",
        "BYOVD (Bring Your Own Vulnerable Driver) attacks for kernel-level persistence",
        "DNS-over-HTTPS tunnelling for command-and-control evasion",
        "MFA fatigue attacks and push notification bombing techniques",
        "Memory-only malware and process injection techniques: reflective DLL, process hollowing",
        "T1190 Exploit Public-Facing Application — top exploited CVEs 2024",
        "Cloud-native attack paths: IAM abuse, SSRF to IMDS, container escape techniques",
        "OT/ICS lateral movement from IT to operational technology networks",
    ],
    "infrastructure": [
        "Nation-state targeting of power grid SCADA and ICS environments",
        "Undersea cable sabotage and maritime infrastructure threat actors 2024-2025",
        "Water treatment facility cyberattacks — Oldsmar and subsequent incidents",
        "Satellite communications security: Viasat KA-SAT attack TTPs and replication risk",
        "Port and logistics infrastructure cyber vulnerabilities — COSCO and DP World incidents",
        "Hospital and healthcare sector ransomware: operational impact and attribution",
        "Telecommunications backbone targeting — BGP hijacking and SS7 exploitation",
        "Nuclear facility cyber security posture — Natanz, Kudankulam precedents",
    ],
    "malware": [
        "BlackCat/ALPHV ransomware-as-a-service affiliate programme and negotiation tactics",
        "Cobalt Strike abuse patterns — malleable C2 profiles and detection signatures",
        "FIN7 custom tooling evolution: Carbanak, Bateleur, POWERTRASH",
        "Industroyer2 and Pipedream/INCONTROLLER ICS malware capabilities",
        "Emotet infrastructure reactivation cycles and payload delivery mechanisms",
        "PlugX persistence via USB replication — global implant distribution mapping",
        "UEFI/BIOS firmware implants: LoJax, MosaicRegressor, CosmicStrand analysis",
        "Brute Ratel C4 red-team tool adoption by threat actors post-Cobalt Strike detection",
    ],
    "doctrine": [
        "Russian offensive cyber doctrine: Gerasimov strategy and information warfare integration",
        "PLA Unit 61398 and China's state-directed cyber espionage strategic framework",
        "Iranian cyber capabilities: IRGC-linked APTs and retaliatory escalation patterns",
        "North Korean cyber operations as sanctions evasion and revenue generation mechanism",
        "Five Eyes intelligence sharing on cyber threats — joint advisory attribution patterns",
        "US Cyber Command Hunt Forward operations and persistent engagement doctrine",
        "NATO Article 5 cyber attribution threshold — collective defence and deterrence posture",
        "Cyber mercenaries and hack-for-hire market: NSO Group, Intellexa, Candiru ecosystem",
    ],
    "vulnerabilities": [
        "Zero-day broker market: Zerodium pricing, government acquisition, exploit lifecycle",
        "Citrix Bleed CVE-2023-4966 exploitation wave — timeline and affected organisations",
        "MOVEit Transfer CVE-2023-34362 mass exploitation — Cl0p campaign analysis",
        "Ivanti Connect Secure zero-day exploitation by nation-state actors 2024",
        "ProxyLogon/ProxyShell Exchange Server exploitation — persistent webshell clusters",
        "Fortinet and Palo Alto Networks perimeter device targeting by state actors",
        "CVE-2024-3400 Palo Alto OS command injection — exploitation chain breakdown",
        "Edge device compromise as persistent initial access — CISA KEV pattern analysis",
    ],
    "financial": [
        "SWIFT network fraud and financial sector cyber heist techniques post-Bangladesh Bank",
        "Crypto exchange hot wallet targeting — technical methods and laundering via mixers",
        "Ransomware payment ecosystem — crypto tracing, OFAC sanctions, negotiation firms",
        "SEC cybersecurity disclosure rules impact on incident reporting and breach timelines",
    ],
}

# Smaller curated set for immediate INTSUM generation via /ingest
INGEST_QUERIES: list[str] = [
    "Conduct a comprehensive threat assessment of Volt Typhoon's pre-positioning in US critical infrastructure — actor profile, known TTPs, affected sectors, defensive recommendations",
    "Analyse the current ransomware ecosystem: top-5 active groups, their infrastructure, negotiation patterns, and sectors at highest risk as of early 2025",
    "Assess nation-state targeting of undersea cables and maritime communications infrastructure — incidents, actors, escalation risk",
    "Profile the Lazarus Group's cryptocurrency theft operations — techniques, laundering chains, estimated haul, and countermeasures",
    "Map the supply chain attack surface: key software dependencies exploited 2020-2025, actor attribution, and systemic risk indicators",
]


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {INGRESS_API_TOKEN}",
        "User-Agent": INGRESS_API_UA,
        "Content-Type": "application/json",
    }


def submit_research(client: httpx.Client, topic: str) -> dict:
    resp = client.post(
        f"{INGRESS_API_URL}/research",
        headers=_headers(),
        json={"topic": topic, "desk": CYBER_DESK, "label": "seed-cyber"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def submit_ingest(client: httpx.Client, query: str) -> dict:
    resp = client.post(
        f"{INGRESS_API_URL}/ingest",
        headers=_headers(),
        json={"query": query, "desk": CYBER_DESK, "label": "seed-cyber", "priority": "normal"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Seed the OSIA cyber desk with intel topics")
    parser.add_argument(
        "--mode",
        choices=["research", "ingest", "both"],
        default="research",
        help="research: background knowledge only; ingest: immediate INTSUMs; both: all",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Only seed topics from this research category (use --list-categories to see options)",
    )
    parser.add_argument("--list-categories", action="store_true", help="Print available research categories and exit")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be submitted without making requests")
    parser.add_argument(
        "--delay",
        type=float,
        default=4.0,
        help="Seconds to wait between requests (default: 4.0). /research is rate-limited to 20/min — keep above 3s.",
    )
    args = parser.parse_args()

    if args.list_categories:
        print("Available research categories:")
        for cat, topics in RESEARCH_TOPICS.items():
            print(f"  {cat:<16} ({len(topics)} topics)")
        return

    if not INGRESS_API_TOKEN:
        logger.error("INGRESS_API_TOKEN not set — run: grep INGRESS_API_TOKEN .env")
        sys.exit(1)

    # Resolve which research topics to submit
    if args.category:
        if args.category not in RESEARCH_TOPICS:
            logger.error("Unknown category '%s'. Use --list-categories.", args.category)
            sys.exit(1)
        research_topics = RESEARCH_TOPICS[args.category]
    else:
        research_topics = [t for topics in RESEARCH_TOPICS.values() for t in topics]

    do_research = args.mode in ("research", "both")
    do_ingest = args.mode in ("ingest", "both")

    total = (len(research_topics) if do_research else 0) + (len(INGEST_QUERIES) if do_ingest else 0)
    logger.info(
        "Seed plan: mode=%s category=%s topics=%d dry_run=%s",
        args.mode,
        args.category or "all",
        total,
        args.dry_run,
    )

    if args.dry_run:
        if do_research:
            print(f"\n-- /research ({len(research_topics)} topics) --")
            for t in research_topics:
                print(f"  {t}")
        if do_ingest:
            print(f"\n-- /ingest ({len(INGEST_QUERIES)} queries) --")
            for q in INGEST_QUERIES:
                print(f"  {q}")
        return

    submitted = skipped = errors = 0

    with httpx.Client() as client:
        if do_research:
            logger.info("--- Submitting %d research topics ---", len(research_topics))
            for i, topic in enumerate(research_topics, 1):
                try:
                    result = submit_research(client, topic)
                    if result.get("queued"):
                        logger.info("[%d/%d] queued  — %s", i, len(research_topics), topic[:80])
                        submitted += 1
                    else:
                        logger.info("[%d/%d] skipped (duplicate) — %s", i, len(research_topics), topic[:80])
                        skipped += 1
                except httpx.HTTPStatusError as e:
                    logger.error("[%d/%d] HTTP %d — %s", i, len(research_topics), e.response.status_code, topic[:60])
                    errors += 1
                except Exception as e:
                    logger.error("[%d/%d] error — %s: %s", i, len(research_topics), topic[:60], e)
                    errors += 1

                if i < len(research_topics):
                    time.sleep(args.delay)

        if do_ingest:
            logger.info("--- Submitting %d ingest queries ---", len(INGEST_QUERIES))
            for i, query in enumerate(INGEST_QUERIES, 1):
                try:
                    result = submit_ingest(client, query)
                    logger.info(
                        "[%d/%d] queued task_id=%s — %s", i, len(INGEST_QUERIES), result.get("task_id", "?"), query[:80]
                    )
                    submitted += 1
                except httpx.HTTPStatusError as e:
                    logger.error("[%d/%d] HTTP %d — %s", i, len(INGEST_QUERIES), e.response.status_code, query[:60])
                    errors += 1
                except Exception as e:
                    logger.error("[%d/%d] error — %s: %s", i, len(INGEST_QUERIES), query[:60], e)
                    errors += 1

                if i < len(INGEST_QUERIES):
                    time.sleep(args.delay)

    logger.info("Done. submitted=%d skipped=%d errors=%d", submitted, skipped, errors)
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()

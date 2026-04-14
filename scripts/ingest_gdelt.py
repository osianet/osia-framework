"""
OSIA GDELT 2.0 Global Events Ingestion

Downloads and ingests events from the GDELT 2.0 project into the 'gdelt-events'
Qdrant collection for Information & Psychological Warfare desk RAG retrieval.

GDELT monitors the world's news media in 100+ languages, encoding events using
the CAMEO taxonomy. Each 15-minute file covers the globe. This script filters
for events relevant across multiple OSIA desks: any event that clears the
min-mentions coverage bar (including cooperation, diplomacy, aid, and conflict),
plus lower-threshold passes for media-actor events and known infowar CAMEO codes.

Source: https://www.gdeltproject.org/
Free public data, no authentication required. Updated every 15 minutes.

CAMEO QuadClass:
  1 = Verbal Cooperation  2 = Material Cooperation
  3 = Verbal Conflict     4 = Material Conflict

Filter (any condition passes):
  NumMentions >= min_mentions (default 10)  — any sufficiently covered event
  Actor is media type "MED" AND NumMentions >= 5  — media-actor events
  CAMEO infowar root code AND NumMentions >= 5  — statements, coercion, protest, etc.

Usage:
  uv run python scripts/ingest_gdelt.py
  uv run python scripts/ingest_gdelt.py --dry-run
  uv run python scripts/ingest_gdelt.py --resume
  uv run python scripts/ingest_gdelt.py --days-back 7
  uv run python scripts/ingest_gdelt.py --enqueue-notable
  uv run python scripts/ingest_gdelt.py --min-mentions 20

Options:
  --dry-run             Parse and embed but skip Qdrant writes and Redis updates
  --resume              Resume from last Redis checkpoint (last processed file timestamp)
  --days-back N         Days to backfill on first run (default: 7)
  --min-mentions N      Minimum NumMentions to include a conflict event (default: 10)
  --enqueue-notable     Push high-coverage conflict events to InfoWar research queue
  --limit N             Stop after N events ingested (0 = no limit)
  --embed-batch-size    Texts per HF embedding call (default: 48)
  --embed-concurrency   Parallel embedding calls (default: 3)
  --upsert-batch-size   Points per Qdrant upsert call (default: 64)

Environment variables (from .env):
  HF_TOKEN              HuggingFace token (required for embeddings)
  QDRANT_URL            Qdrant URL (default: https://qdrant.osia.dev)
  QDRANT_API_KEY        Qdrant API key
  REDIS_URL             Redis URL (default: redis://localhost:6379)
"""

import argparse
import asyncio
import csv
import hashlib
import io
import json
import logging
import math
import os
import re
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import httpx
import redis.asyncio as aioredis
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("osia.gdelt_ingest")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN", "")
QDRANT_URL = os.getenv("QDRANT_URL", "https://qdrant.osia.dev")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "") or None
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

COLLECTION_NAME = "gdelt-events"
EMBEDDING_DIM = 384
SOURCE_LABEL = "GDELT 2.0 Global Event Database"

GDELT_MASTERLIST = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
USER_AGENT = "OSIA-Framework/1.0 (open-source intelligence research; +https://osia.dev)"
REQUEST_DELAY = 1.0

HF_EMBEDDING_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
)

CHECKPOINT_KEY = "osia:gdelt:last_ts"
RESEARCH_QUEUE_KEY = "osia:research_queue"

TODAY = datetime.now(UTC).strftime("%Y-%m-%d")

# GDELT 2.0 export CSV column names (no header in file, 61 columns)
GDELT_COLUMNS = [
    "GlobalEventID",
    "Day",
    "MonthYear",
    "Year",
    "FractionDate",
    "Actor1Code",
    "Actor1Name",
    "Actor1CountryCode",
    "Actor1KnownGroupCode",
    "Actor1EthnicCode",
    "Actor1Religion1Code",
    "Actor1Religion2Code",
    "Actor1Type1Code",
    "Actor1Type2Code",
    "Actor1Type3Code",
    "Actor2Code",
    "Actor2Name",
    "Actor2CountryCode",
    "Actor2KnownGroupCode",
    "Actor2EthnicCode",
    "Actor2Religion1Code",
    "Actor2Religion2Code",
    "Actor2Type1Code",
    "Actor2Type2Code",
    "Actor2Type3Code",
    "IsRootEvent",
    "EventCode",
    "EventBaseCode",
    "EventRootCode",
    "QuadClass",
    "GoldsteinScale",
    "NumMentions",
    "NumSources",
    "NumArticles",
    "AvgTone",
    "Actor1Geo_Type",
    "Actor1Geo_FullName",
    "Actor1Geo_CountryCode",
    "Actor1Geo_ADM1Code",
    "Actor1Geo_ADM2Code",
    "Actor1Geo_Lat",
    "Actor1Geo_Long",
    "Actor1Geo_FeatureID",
    "Actor2Geo_Type",
    "Actor2Geo_FullName",
    "Actor2Geo_CountryCode",
    "Actor2Geo_ADM1Code",
    "Actor2Geo_ADM2Code",
    "Actor2Geo_Lat",
    "Actor2Geo_Long",
    "Actor2Geo_FeatureID",
    "ActionGeo_Type",
    "ActionGeo_FullName",
    "ActionGeo_CountryCode",
    "ActionGeo_ADM1Code",
    "ActionGeo_ADM2Code",
    "ActionGeo_Lat",
    "ActionGeo_Long",
    "ActionGeo_FeatureID",
    "DATEADDED",
    "SOURCEURL",
]

# CAMEO event root codes relevant to information warfare
INFOWAR_ROOT_CODES = {"01", "02", "10", "11", "12", "13", "14", "17"}

# QuadClass >= 3 = verbal/material conflict
MIN_QUADCLASS_CONFLICT = 3

# Notable: high media coverage + material conflict
NOTABLE_MIN_MENTIONS = 50
NOTABLE_MIN_QUADCLASS = 4

CHUNK_SIZE = 400
CHUNK_OVERLAP_WORDS = 50

# GDELT uses FIPS 10-4 country codes (NOT ISO 3166).  Many differ from ISO:
# CH=China (not Switzerland), GM=Germany (not Gambia), RS=Russia (not Serbia),
# AS=Australia (not American Samoa), SF=South Africa, UP=Ukraine, etc.
FIPS_COUNTRY_NAMES: dict[str, str] = {
    "AA": "Aruba",
    "AC": "Antigua and Barbuda",
    "AF": "Afghanistan",
    "AG": "Algeria",
    "AJ": "Azerbaijan",
    "AL": "Albania",
    "AM": "Armenia",
    "AN": "Andorra",
    "AO": "Angola",
    "AR": "Argentina",
    "AS": "Australia",
    "AU": "Austria",
    "AV": "Anguilla",
    "AY": "Antarctica",
    "BA": "Bahrain",
    "BB": "Barbados",
    "BC": "Botswana",
    "BE": "Belgium",
    "BF": "Bahamas",
    "BG": "Bangladesh",
    "BH": "Belize",
    "BK": "Bosnia and Herzegovina",
    "BL": "Bolivia",
    "BM": "Burma (Myanmar)",
    "BN": "Benin",
    "BO": "Belarus",
    "BP": "Solomon Islands",
    "BR": "Brazil",
    "BT": "Bhutan",
    "BU": "Bulgaria",
    "BY": "Burundi",
    "CA": "Canada",
    "CB": "Cambodia",
    "CD": "Chad",
    "CE": "Sri Lanka",
    "CF": "Congo (Republic)",
    "CG": "Congo (DRC)",
    "CH": "China",
    "CI": "Chile",
    "CM": "Cameroon",
    "CN": "Comoros",
    "CO": "Colombia",
    "CS": "Costa Rica",
    "CT": "Central African Republic",
    "CU": "Cuba",
    "CV": "Cape Verde",
    "CY": "Cyprus",
    "DA": "Denmark",
    "DJ": "Djibouti",
    "DO": "Dominica",
    "DR": "Dominican Republic",
    "EC": "Ecuador",
    "EG": "Egypt",
    "EI": "Ireland",
    "EK": "Equatorial Guinea",
    "EN": "Estonia",
    "ER": "Eritrea",
    "ES": "El Salvador",
    "ET": "Ethiopia",
    "EZ": "Czech Republic",
    "FI": "Finland",
    "FJ": "Fiji",
    "FK": "Falkland Islands",
    "FM": "Micronesia",
    "FO": "Faroe Islands",
    "FR": "France",
    "GA": "Gambia",
    "GB": "Gabon",
    "GG": "Georgia",
    "GH": "Ghana",
    "GJ": "Grenada",
    "GL": "Greenland",
    "GM": "Germany",
    "GO": "Gabon",
    "GP": "Guadeloupe",
    "GR": "Greece",
    "GT": "Guatemala",
    "GV": "Guinea",
    "GY": "Guyana",
    "GZ": "Gaza Strip",
    "HA": "Haiti",
    "HK": "Hong Kong",
    "HO": "Honduras",
    "HR": "Croatia",
    "HU": "Hungary",
    "IC": "Iceland",
    "ID": "Indonesia",
    "IN": "India",
    "IR": "Iran",
    "IS": "Israel",
    "IT": "Italy",
    "IV": "Côte d'Ivoire",
    "IZ": "Iraq",
    "JA": "Japan",
    "JM": "Jamaica",
    "JO": "Jordan",
    "KE": "Kenya",
    "KG": "Kyrgyzstan",
    "KN": "North Korea",
    "KS": "South Korea",
    "KU": "Kuwait",
    "KV": "Kosovo",
    "KZ": "Kazakhstan",
    "LA": "Laos",
    "LE": "Lebanon",
    "LG": "Latvia",
    "LI": "Liberia",
    "LO": "Slovakia",
    "LT": "Lithuania",
    "LU": "Luxembourg",
    "LY": "Libya",
    "MA": "Madagascar",
    "MC": "Macau",
    "MD": "Moldova",
    "MG": "Mongolia",
    "MI": "Malawi",
    "MJ": "Montenegro",
    "MK": "North Macedonia",
    "ML": "Mali",
    "MM": "Malta",
    "MN": "Monaco",
    "MO": "Morocco",
    "MP": "Mauritius",
    "MR": "Mauritania",
    "MU": "Oman",
    "MV": "Maldives",
    "MX": "Mexico",
    "MY": "Malaysia",
    "MZ": "Mozambique",
    "NG": "Niger",
    "NH": "Vanuatu",
    "NI": "Nigeria",
    "NL": "Netherlands",
    "NO": "Norway",
    "NP": "Nepal",
    "NR": "Nauru",
    "NS": "Suriname",
    "NU": "Nicaragua",
    "NZ": "New Zealand",
    "PA": "Paraguay",
    "PE": "Peru",
    "PK": "Pakistan",
    "PL": "Poland",
    "PM": "Panama",
    "PO": "Portugal",
    "PP": "Papua New Guinea",
    "PS": "Palau",
    "PU": "Guinea-Bissau",
    "QA": "Qatar",
    "RI": "Serbia",
    "RM": "Marshall Islands",
    "RO": "Romania",
    "RP": "Philippines",
    "RS": "Russia",
    "RW": "Rwanda",
    "SA": "Saudi Arabia",
    "SC": "Saint Kitts and Nevis",
    "SE": "Seychelles",
    "SF": "South Africa",
    "SG": "Senegal",
    "SI": "Slovenia",
    "SL": "Sierra Leone",
    "SM": "San Marino",
    "SN": "Singapore",
    "SO": "Somalia",
    "SP": "Spain",
    "SS": "South Sudan",
    "ST": "Saint Lucia",
    "SU": "Sudan",
    "SW": "Sweden",
    "SY": "Syria",
    "SZ": "Switzerland",
    "TD": "Trinidad and Tobago",
    "TH": "Thailand",
    "TI": "Tajikistan",
    "TN": "Tonga",
    "TO": "Togo",
    "TS": "Tunisia",
    "TT": "Timor-Leste",
    "TU": "Turkey",
    "TV": "Tuvalu",
    "TW": "Taiwan",
    "TX": "Turkmenistan",
    "TZ": "Tanzania",
    "UG": "Uganda",
    "UK": "United Kingdom",
    "UP": "Ukraine",
    "US": "United States",
    "UV": "Burkina Faso",
    "UY": "Uruguay",
    "UZ": "Uzbekistan",
    "VC": "Saint Vincent and the Grenadines",
    "VE": "Venezuela",
    "VM": "Vietnam",
    "WA": "Namibia",
    "WE": "West Bank",
    "WS": "Samoa",
    "WZ": "Eswatini",
    "YM": "Yemen",
    "ZA": "Zambia",
    "ZI": "Zimbabwe",
}

# GDELT actor type codes → human-readable labels (used in narrative and payload)
ACTOR_TYPE_NAMES: dict[str, str] = {
    "GOV": "government",
    "MIL": "military",
    "MED": "media",
    "IGO": "intergovernmental organization",
    "NGO": "NGO",
    "BUS": "business/corporate",
    "COP": "police/law enforcement",
    "CVL": "civilian",
    "CIV": "civilian",
    "OPP": "opposition group",
    "JUD": "judiciary",
    "LAB": "labour group",
    "AGR": "agricultural sector",
    "EDU": "education sector",
    "HLH": "health sector",
    "LEG": "legislature",
    "REL": "religious group",
    "SPY": "intelligence services",
    "SOC": "social movement",
    "REB": "rebel/insurgent group",
    "MOB": "mob/crowd",
    "REF": "refugee/displaced persons",
    "IMM": "immigrant community",
    "ACT": "activist",
    "MIS": "paramilitary",
    "SET": "settler",
    "TRO": "troops/armed forces",
    "UNK": "unknown actor",
}

# CAMEO root code → short descriptive label (used in structured fields)
CAMEO_ROOT_DESC: dict[str, str] = {
    "01": "Public Statement",
    "02": "Appeal",
    "03": "Intent to Cooperate",
    "04": "Diplomatic Consultation",
    "05": "Diplomatic Cooperation",
    "06": "Material Cooperation",
    "07": "Aid / Assistance",
    "08": "Yield / Concede",
    "09": "Investigation",
    "10": "Demand",
    "11": "Disapproval / Criticism",
    "12": "Rejection",
    "13": "Threat",
    "14": "Protest / Demonstration",
    "15": "Military Force Posture",
    "16": "Reduce / Sever Relations",
    "17": "Coercion",
    "18": "Assault",
    "19": "Armed Conflict",
    "20": "Mass Violence",
}

# mentions_weight normalisation ceiling — log1p(200) ≈ 5.3
# Events with 200+ mentions score 1.0; events at the 10-mention floor score ~0.45
_MENTIONS_NORM_CEIL = math.log1p(200)


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP_WORDS) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if not text:
        return []
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return [c for c in chunks if len(c.strip()) >= 60]


def _parse_file_ts(url: str) -> str | None:
    """Extract YYYYMMDDHHMMSS timestamp string from a GDELT file URL."""
    m = re.search(r"/(\d{14})\.export\.CSV\.zip", url)
    return m.group(1) if m else None


def _ts_to_unix(ts_str: str) -> int:
    """Convert GDELT timestamp string YYYYMMDDHHMMSS to Unix timestamp."""
    dt = datetime.strptime(ts_str, "%Y%m%d%H%M%S")
    return int(dt.replace(tzinfo=UTC).timestamp())


def _date_to_ts_prefix(dt: datetime) -> str:
    """Return a YYYYMMDD prefix for filtering masterlist entries by date."""
    return dt.strftime("%Y%m%d")


# ---------------------------------------------------------------------------
# Document builder
# ---------------------------------------------------------------------------


# CAMEO root code → natural-language verb phrase for narrative sentences.
# Covers all 20 root codes; used by build_document() to produce prose the
# sentence-transformer can embed effectively.
_CAMEO_VERB: dict[str, str] = {
    "01": "made a public statement regarding",
    "02": "appealed to",
    "03": "expressed intent to cooperate with",
    "04": "consulted with",
    "05": "engaged in diplomatic cooperation with",
    "06": "provided material cooperation to",
    "07": "provided aid or assistance to",
    "08": "yielded or conceded to",
    "09": "investigated",
    "10": "issued demands toward",
    "11": "expressed disapproval or criticism of",
    "12": "rejected or refused",
    "13": "issued threats against",
    "14": "staged protests or demonstrations against",
    "15": "exhibited military force posture toward",
    "16": "reduced or severed relations with",
    "17": "used coercive measures against",
    "18": "carried out an assault on",
    "19": "engaged in armed fighting with",
    "20": "committed mass violence against",
}


def _country_name(fips: str) -> str:
    """Resolve a FIPS country code to its full name, falling back to the code itself."""
    return FIPS_COUNTRY_NAMES.get(fips.strip().upper(), fips) if fips else ""


def _actor_label(name: str, fips_country: str, type_code: str) -> str:
    """Build a human-readable actor label with full country name and type."""
    parts = [name] if name else []
    country = _country_name(fips_country)
    if country and country not in (name or ""):
        parts.append(f"({country})")
    actor_type = ACTOR_TYPE_NAMES.get(type_code.strip().upper(), "") if type_code else ""
    if actor_type:
        parts.append(f"[{actor_type}]")
    return " ".join(parts)


def _narrative_sentence(
    actor1_name: str,
    actor1_country: str,
    actor1_type: str,
    actor2_name: str,
    actor2_country: str,
    actor2_type: str,
    event_root: str,
    quad_label: str,
    action_geo: str,
    date_str: str,
    num_mentions: str,
    avg_tone: str,
    goldstein: str,
) -> str:
    """Produce prose sentences summarising the GDELT event for the embedding model."""
    root = str(event_root or "").zfill(2)
    verb = _CAMEO_VERB.get(root, f"was involved in a {quad_label.lower()} event with")

    # Subject: name + full country name + actor role
    a1_country = _country_name(actor1_country)
    subject = actor1_name or "An unidentified actor"
    if a1_country and a1_country not in subject:
        subject += f" ({a1_country})"
    a1_role = ACTOR_TYPE_NAMES.get((actor1_type or "").strip().upper(), "")
    if a1_role:
        subject += f", a {a1_role},"

    # Object: same pattern
    obj = ""
    if actor2_name:
        a2_country = _country_name(actor2_country)
        obj = actor2_name
        if a2_country and a2_country not in obj:
            obj += f" ({a2_country})"
        a2_role = ACTOR_TYPE_NAMES.get((actor2_type or "").strip().upper(), "")
        if a2_role:
            obj += f" [{a2_role}]"

    geo_part = f" in {action_geo}" if action_geo else ""
    date_part = f" on {date_str}" if date_str else ""

    sentence = f"{subject} {verb}"
    if obj:
        sentence += f" {obj}"
    sentence += f"{geo_part}{date_part}."

    # Coverage descriptor — lets queries like "widely reported" or "major incident" match
    try:
        nm = int(num_mentions)
        if nm >= 100:
            sentence += f" This was a widely reported major event ({nm} media mentions)."
        elif nm >= 50:
            sentence += f" This event received significant media coverage ({nm} mentions)."
        elif nm >= 20:
            sentence += f" This event received moderate media coverage ({nm} mentions)."
        else:
            sentence += f" This event received limited media coverage ({nm} mentions)."
    except (ValueError, TypeError):
        pass

    # Tone descriptor
    try:
        tone = float(avg_tone)
        if tone < -10:
            sentence += " Media coverage was highly hostile in tone."
        elif tone < -5:
            sentence += " Media coverage was hostile in tone."
        elif tone < 0:
            sentence += " Media coverage had a negative tone."
        else:
            sentence += " Media coverage had a neutral or positive tone."
    except (ValueError, TypeError):
        pass

    # Goldstein stability context
    try:
        gs = float(goldstein)
        if gs <= -7:
            sentence += " This event is highly destabilising for regional stability."
        elif gs <= -3:
            sentence += " This event has a destabilising effect on regional stability."
        elif gs >= 7:
            sentence += " This event is highly stabilising."
        elif gs >= 3:
            sentence += " This event has a stabilising effect."
    except (ValueError, TypeError):
        pass

    return sentence


def build_document(row: dict) -> tuple[str, int | None]:
    """Build a narrative + structured document from a GDELT event row."""
    event_id = row.get("GlobalEventID", "")
    day = row.get("Day", "")
    actor1_name = row.get("Actor1Name", "") or ""
    actor1_country = row.get("Actor1CountryCode", "") or ""
    actor1_type = row.get("Actor1Type1Code", "") or ""
    actor2_name = row.get("Actor2Name", "") or ""
    actor2_country = row.get("Actor2CountryCode", "") or ""
    actor2_type = row.get("Actor2Type1Code", "") or ""
    event_code = row.get("EventCode", "") or ""
    event_root = row.get("EventRootCode", "") or ""
    quad_class = row.get("QuadClass", "") or ""
    goldstein = row.get("GoldsteinScale", "") or ""
    num_mentions = row.get("NumMentions", "") or ""
    num_articles = row.get("NumArticles", "") or ""
    avg_tone = row.get("AvgTone", "") or ""
    action_geo = row.get("ActionGeo_FullName", "") or ""
    action_country = row.get("ActionGeo_CountryCode", "") or ""
    source_url = row.get("SOURCEURL", "") or ""

    # Parse event date from Day field (YYYYMMDD)
    event_unix: int | None = None
    if day and len(day) == 8:
        try:
            dt = datetime.strptime(day, "%Y%m%d")
            event_unix = int(dt.replace(tzinfo=UTC).timestamp())
        except ValueError:
            pass  # malformed YYYYMMDD date in GDELT row — leave event_unix as None

    if not (actor1_name or actor2_name or action_geo):
        return "", event_unix

    quad_labels = {
        "1": "Verbal Cooperation",
        "2": "Material Cooperation",
        "3": "Verbal Conflict",
        "4": "Material Conflict",
    }
    quad_label = quad_labels.get(str(quad_class), f"QuadClass {quad_class}")

    date_str = f"{day[:4]}-{day[4:6]}-{day[6:8]}" if day and len(day) == 8 else day

    root = str(event_root).zfill(2)
    cameo_desc = CAMEO_ROOT_DESC.get(root, f"Event root {event_root}")
    action_country_name = _country_name(action_country)

    lines: list[str] = []

    # Lead with prose — gives the embedding model natural language to anchor on
    narrative = _narrative_sentence(
        actor1_name,
        actor1_country,
        actor1_type,
        actor2_name,
        actor2_country,
        actor2_type,
        event_root,
        quad_label,
        action_geo,
        date_str,
        num_mentions,
        avg_tone,
        goldstein,
    )
    lines.append(narrative)
    lines.append("")

    # Structured fields — retain FIPS codes alongside full names for completeness
    lines.append(f"Event Type: {cameo_desc} — GDELT {quad_label} (CAMEO {event_code} / root {event_root})")
    if date_str:
        lines.append(f"Date: {date_str}")

    if actor1_name:
        lines.append(f"Actor 1: {_actor_label(actor1_name, actor1_country, actor1_type)}")
    if actor2_name:
        lines.append(f"Actor 2: {_actor_label(actor2_name, actor2_country, actor2_type)}")

    if action_geo:
        loc = action_geo
        if action_country_name and action_country_name not in action_geo:
            loc += f", {action_country_name}"
        if action_country and action_country not in action_geo:
            loc += f" [{action_country}]"
        lines.append(f"Location: {loc}")

    try:
        nm, na = int(num_mentions), int(num_articles)
        lines.append(f"Media Coverage: {nm} mentions across {na} articles")
    except (ValueError, TypeError):
        pass

    try:
        tone_val = float(avg_tone)
        if tone_val < -10:
            tone_desc = "highly hostile"
        elif tone_val < -5:
            tone_desc = "hostile"
        elif tone_val < 0:
            tone_desc = "negative"
        else:
            tone_desc = "positive"
        lines.append(f"Average Tone: {tone_val:.2f} ({tone_desc})")
    except (ValueError, TypeError):
        pass

    try:
        gs = float(goldstein)
        if gs <= -7:
            gs_desc = "highly destabilising"
        elif gs <= -3:
            gs_desc = "destabilising"
        elif gs < 0:
            gs_desc = "mildly destabilising"
        elif gs < 3:
            gs_desc = "neutral"
        elif gs < 7:
            gs_desc = "stabilising"
        else:
            gs_desc = "highly stabilising"
        lines.append(f"Goldstein Scale: {gs:.1f} ({gs_desc})")
    except (ValueError, TypeError):
        pass

    if source_url:
        lines.append(f"Source: {source_url}")
    if event_id:
        lines.append(f"GDELT Event ID: {event_id}")

    if len(lines) < 4:
        return "", event_unix

    return "\n".join(lines), event_unix


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class IngestStats:
    files_processed: int = 0
    records_seen: int = 0
    records_skipped: int = 0
    records_processed: int = 0
    points_upserted: int = 0
    events_enqueued: int = 0
    errors: int = 0
    started_at: float = field(default_factory=time.time)

    def elapsed(self) -> str:
        secs = int(time.time() - self.started_at)
        return f"{secs // 60}m{secs % 60:02d}s"

    def log_progress(self) -> None:
        logger.info(
            "files=%d seen=%d processed=%d skipped=%d upserted=%d enqueued=%d errors=%d elapsed=%s",
            self.files_processed,
            self.records_seen,
            self.records_processed,
            self.records_skipped,
            self.points_upserted,
            self.events_enqueued,
            self.errors,
            self.elapsed(),
        )


# ---------------------------------------------------------------------------
# Main ingestor
# ---------------------------------------------------------------------------


class GdeltIngestor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.dry_run: bool = args.dry_run
        self.enqueue_notable: bool = args.enqueue_notable
        self.limit: int = args.limit
        self.days_back: int = args.days_back
        self.min_mentions: int = args.min_mentions
        self.embed_batch_size: int = args.embed_batch_size
        self.embed_concurrency: int = args.embed_concurrency
        self.upsert_batch_size: int = args.upsert_batch_size
        self.upsert_delay: float = args.upsert_delay
        self.resume: bool = args.resume

        self._qdrant = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, port=None)
        self._redis: aioredis.Redis | None = None
        self._embed_semaphore = asyncio.Semaphore(self.embed_concurrency)
        self._upsert_buffer: list[qdrant_models.PointStruct] = []

    async def run(self) -> None:
        self._redis = aioredis.from_url(REDIS_URL, decode_responses=True)
        try:
            await self._ensure_collection()

            cutoff_ts, from_checkpoint = await self._resolve_cutoff_ts()
            logger.info(
                "Processing GDELT files newer than ts=%s (source=%s)",
                cutoff_ts or "none (all)",
                "checkpoint" if from_checkpoint else "days-back",
            )

            stats = IngestStats()
            file_urls = await self._get_file_urls(cutoff_ts, from_checkpoint=from_checkpoint)
            logger.info("Found %d GDELT files to process", len(file_urls))

            async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}, timeout=120.0) as http:
                for url, ts_str in file_urls:
                    try:
                        await self._process_file(url, ts_str, stats, http)
                        await self._save_checkpoint(ts_str)
                        stats.log_progress()
                    except Exception as exc:
                        stats.errors += 1
                        logger.error("Failed to process %s: %s", url, exc)

                    if self.limit and stats.records_processed >= self.limit:
                        logger.info("Reached --limit %d — stopping.", self.limit)
                        break

                    await asyncio.sleep(REQUEST_DELAY)

            await self._flush_upsert_buffer(stats)
            stats.log_progress()
            logger.info("GDELT ingestion complete.")
        finally:
            await self._qdrant.close()
            if self._redis:
                await self._redis.aclose()

    async def _resolve_cutoff_ts(self) -> tuple[str | None, bool]:
        """Return (cutoff_ts, from_checkpoint).

        from_checkpoint=True means the cutoff came from a saved Redis checkpoint,
        so we can generate file URLs directly (no masterlist download needed).
        """
        if self.resume and self._redis:
            checkpoint = await self._redis.get(CHECKPOINT_KEY)
            if checkpoint:
                logger.info("Resuming from checkpoint ts=%s", checkpoint)
                return checkpoint, True
            logger.warning("No Redis checkpoint found for --resume; falling back to --days-back %d", self.days_back)
        cutoff_dt = datetime.now(UTC) - timedelta(days=self.days_back)
        return cutoff_dt.strftime("%Y%m%d%H%M%S"), False

    async def _get_file_urls(self, cutoff_ts: str | None, from_checkpoint: bool = False) -> list[tuple[str, str]]:
        """Return (url, ts_str) pairs for GDELT files newer than cutoff_ts.

        When from_checkpoint=True the cutoff is recent and we generate URLs
        directly from the 15-minute GDELT file schedule, avoiding the ~50 MB
        masterfilelist.txt download entirely.
        """
        if from_checkpoint and cutoff_ts:
            return self._generate_urls_since(cutoff_ts)

        # Full backfill path: download masterfilelist.txt and filter.
        async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}, timeout=60.0) as http:
            for attempt in range(3):
                try:
                    resp = await http.get(GDELT_MASTERLIST)
                    resp.raise_for_status()
                    break
                except Exception as exc:
                    logger.warning("Masterlist fetch attempt %d failed: %s", attempt + 1, exc)
                    await asyncio.sleep(10 * (attempt + 1))
            else:
                logger.error("Failed to fetch GDELT masterlist.")
                return []

        results: list[tuple[str, str]] = []
        for line in resp.text.splitlines():
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            url = parts[2]
            if ".export.CSV.zip" not in url:
                continue
            ts_str = _parse_file_ts(url)
            if not ts_str:
                continue
            if cutoff_ts and ts_str <= cutoff_ts:
                continue
            results.append((url, ts_str))

        results.sort(key=lambda x: x[1])
        return results

    @staticmethod
    def _generate_urls_since(cutoff_ts: str) -> list[tuple[str, str]]:
        """Generate GDELT export URLs on 15-minute boundaries from cutoff to now.

        GDELT 2.0 files are published as:
          http://data.gdeltproject.org/gdeltv2/{YYYYMMDDHHMMSS}.export.CSV.zip
        at :00, :15, :30, :45 past each hour, so we can derive the full list
        without touching the masterfilelist.
        """
        cutoff_dt = datetime.strptime(cutoff_ts, "%Y%m%d%H%M%S").replace(tzinfo=UTC)
        now = datetime.now(UTC)

        # Snap to the first 15-minute slot strictly after the checkpoint.
        next_slot_min = (cutoff_dt.minute // 15 + 1) * 15
        current = cutoff_dt.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=next_slot_min)

        results: list[tuple[str, str]] = []
        while current <= now:
            ts_str = current.strftime("%Y%m%d%H%M%S")
            url = f"http://data.gdeltproject.org/gdeltv2/{ts_str}.export.CSV.zip"
            results.append((url, ts_str))
            current += timedelta(minutes=15)

        logger.info("Generated %d direct GDELT file URLs from %s to now.", len(results), cutoff_ts)
        return results

    async def _process_file(self, url: str, ts_str: str, stats: IngestStats, http: httpx.AsyncClient) -> None:
        logger.info("Processing GDELT file %s", ts_str)

        for attempt in range(4):
            try:
                resp = await http.get(url)
                if resp.status_code == 404:
                    # File not yet published or no longer available — skip silently.
                    logger.debug("GDELT file not available (404): %s", ts_str)
                    stats.files_processed += 1
                    return
                if resp.status_code == 429:
                    await asyncio.sleep(30 * (attempt + 1))
                    continue
                resp.raise_for_status()
                zip_bytes = resp.content
                break
            except Exception as exc:
                logger.warning("Download attempt %d failed for %s: %s", attempt + 1, ts_str, exc)
                await asyncio.sleep(10 * (attempt + 1))
        else:
            raise RuntimeError(f"Failed to download {url}")

        rows = await asyncio.get_event_loop().run_in_executor(None, self._parse_zip_csv, zip_bytes)

        file_processed = 0
        for row in rows:
            stats.records_seen += 1
            if not self._passes_filter(row):
                stats.records_skipped += 1
                continue

            try:
                await self._process_row(row, ts_str, stats)
                file_processed += 1
            except Exception as exc:
                stats.errors += 1
                logger.debug("Row error: %s", exc)

            if self.limit and stats.records_processed >= self.limit:
                break

        await self._flush_upsert_buffer(stats)
        stats.files_processed += 1
        logger.debug("File %s: %d rows kept of %d seen", ts_str, file_processed, len(rows))

    def _parse_zip_csv(self, zip_bytes: bytes) -> list[dict]:
        """Decompress and parse a GDELT 2.0 export CSV zip."""
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = [n for n in zf.namelist() if n.endswith(".CSV")]
            if not names:
                return []
            with zf.open(names[0]) as fh:
                text = fh.read().decode("utf-8", errors="replace")
        reader = csv.DictReader(
            io.StringIO(text),
            fieldnames=GDELT_COLUMNS,
            delimiter="\t",
        )
        return list(reader)

    def _passes_filter(self, row: dict) -> bool:
        """Return True if this event is worth ingesting for any OSIA desk.

        The collection serves InfoWar, Geopolitical, Finance, HUMINT, Environment,
        and Watch Floor desks — not just InfoWar.  The primary gate is media
        coverage: any event that cleared the min_mentions bar has been deemed
        newsworthy enough to appear in global news at scale.  QuadClass 1-2
        (cooperation, aid, diplomacy) are explicitly included because they are
        intelligence-relevant for Geopolitical and Finance desks.  Lower
        thresholds are kept for media-actor events and known infowar CAMEO codes.
        """
        try:
            num_mentions = int(row.get("NumMentions", 0) or 0)
        except (ValueError, TypeError):
            return False

        actor1_type = row.get("Actor1Type1Code", "") or ""
        actor2_type = row.get("Actor2Type1Code", "") or ""
        event_root = str(row.get("EventRootCode", "") or "").zfill(2)

        # Primary gate: any sufficiently covered event, regardless of QuadClass.
        is_high_coverage = num_mentions >= self.min_mentions

        # Lower threshold for media actors and known infowar CAMEO codes.
        is_media_event = ("MED" in (actor1_type, actor2_type)) and num_mentions >= 5
        is_infowar_code = event_root in INFOWAR_ROOT_CODES and num_mentions >= 5

        return is_high_coverage or is_media_event or is_infowar_code

    async def _process_row(self, row: dict, ts_str: str, stats: IngestStats) -> None:
        event_id = row.get("GlobalEventID", "")
        doc, event_unix = build_document(row)
        if not doc.strip():
            stats.records_skipped += 1
            return

        stats.records_processed += 1

        # --- Numeric fields ---
        try:
            quad_class = int(row.get("QuadClass", 0) or 0)
        except (ValueError, TypeError):
            quad_class = 0
        try:
            num_mentions = int(row.get("NumMentions", 0) or 0)
        except (ValueError, TypeError):
            num_mentions = 0
        try:
            avg_tone = round(float(row.get("AvgTone", 0.0) or 0.0), 3)
        except (ValueError, TypeError):
            avg_tone = 0.0
        try:
            goldstein_scale = round(float(row.get("GoldsteinScale", 0.0) or 0.0), 2)
        except (ValueError, TypeError):
            goldstein_scale = 0.0

        # --- String fields ---
        actor1_name = (row.get("Actor1Name", "") or "").strip()
        actor1_fips = (row.get("Actor1CountryCode", "") or "").strip()
        actor1_type = (row.get("Actor1Type1Code", "") or "").strip()
        actor2_name = (row.get("Actor2Name", "") or "").strip()
        actor2_fips = (row.get("Actor2CountryCode", "") or "").strip()
        actor2_type = (row.get("Actor2Type1Code", "") or "").strip()
        action_country_fips = (row.get("ActionGeo_CountryCode", "") or "").strip()
        action_geo = (row.get("ActionGeo_FullName", "") or "").strip()
        event_root_code = (row.get("EventRootCode", "") or "").strip().zfill(2)
        source_url = (row.get("SOURCEURL", "") or "").strip()
        day = (row.get("Day", "") or "").strip()

        # --- Resolved names ---
        actor1_country_name = _country_name(actor1_fips)
        actor2_country_name = _country_name(actor2_fips)
        action_country_name = _country_name(action_country_fips)

        pub_date = f"{day[:4]}-{day[4:6]}-{day[6:8]}" if len(day) == 8 else ""
        ingest_unix = event_unix or int(time.time())

        # mentions_weight: log-normalised [0, 1] for score boosting at query time.
        # Ceiling at 200 mentions → 1.0; 10 mentions → ~0.45; 1 mention → ~0.13
        mentions_weight = round(min(math.log1p(num_mentions) / _MENTIONS_NORM_CEIL, 1.0), 4)

        # entity_tags: full names only — no FIPS codes, no CAMEO codes.
        # Used by cross_desk_search entity filtering and for human readability.
        raw_tags = [
            actor1_name,
            actor1_country_name,
            actor2_name,
            actor2_country_name,
            action_geo,
            action_country_name,
        ]
        entity_tags = list(dict.fromkeys(t for t in raw_tags if t))

        point_id = str(uuid.UUID(bytes=hashlib.sha256(f"gdelt:{event_id}:{ts_str}".encode()).digest()[:16]))
        payload: dict = {
            "text": doc,
            "source": SOURCE_LABEL,
            "document_type": "global_event",
            "provenance": "gdelt2",
            "ingest_date": TODAY,
            "event_id": event_id,
            "file_ts": ts_str,
            # Event classification
            "quad_class": quad_class,
            "event_root_code": event_root_code,
            "cameo_description": CAMEO_ROOT_DESC.get(event_root_code, ""),
            # Actors — both name and resolved country name; FIPS retained for reference
            "actor1_name": actor1_name,
            "actor1_country": actor1_fips,
            "actor1_country_name": actor1_country_name,
            "actor1_type": ACTOR_TYPE_NAMES.get(actor1_type.upper(), actor1_type),
            "actor2_name": actor2_name,
            "actor2_country": actor2_fips,
            "actor2_country_name": actor2_country_name,
            "actor2_type": ACTOR_TYPE_NAMES.get(actor2_type.upper(), actor2_type),
            # Geography
            "country": action_country_fips,  # FIPS — retained for backward compat
            "country_name": action_country_name,  # full name — primary for display/filter
            "action_geo": action_geo,
            # Metrics
            "num_mentions": num_mentions,
            "avg_tone": avg_tone,
            "goldstein_scale": goldstein_scale,
            "mentions_weight": mentions_weight,
            # Temporal
            "pub_date": pub_date,
            "ingested_at_unix": ingest_unix,
            # Provenance
            "source_url": source_url,
            "entity_tags": entity_tags,
        }

        self._upsert_buffer.append(
            qdrant_models.PointStruct(id=point_id, vector=[0.0] * EMBEDDING_DIM, payload=payload)
        )

        if len(self._upsert_buffer) >= self.upsert_batch_size:
            await self._flush_upsert_buffer(stats)

        if self.enqueue_notable and quad_class >= NOTABLE_MIN_QUADCLASS and num_mentions >= NOTABLE_MIN_MENTIONS:
            await self._maybe_enqueue(event_id, row, stats)

    async def _maybe_enqueue(self, event_id: str, row: dict, stats: IngestStats) -> None:
        if not self._redis or self.dry_run:
            return
        redis_key = f"osia:gdelt:enqueued:{event_id}"
        if await self._redis.exists(redis_key):
            return

        actor1 = row.get("Actor1Name", "") or ""
        actor2 = row.get("Actor2Name", "") or ""
        action_geo = row.get("ActionGeo_FullName", "") or ""
        num_mentions = row.get("NumMentions", "")
        source_url = row.get("SOURCEURL", "") or ""
        day = row.get("Day", "") or ""
        pub_date = f"{day[:4]}-{day[4:6]}-{day[6:8]}" if len(day) == 8 else day

        topic = f"GDELT high-impact conflict: {actor1} vs {actor2} in {action_geo} ({pub_date})"
        topic += f" — {num_mentions} media mentions"
        if source_url:
            topic += f" — source: {source_url[:120]}"

        job = json.dumps(
            {
                "job_id": str(uuid.uuid4()),
                "topic": topic,
                "desk": "information-warfare-desk",
                "priority": "normal",
                "triggered_by": "gdelt_ingest",
                "metadata": {
                    "event_id": event_id,
                    "actor1": actor1,
                    "actor2": actor2,
                    "location": action_geo,
                    "source_url": source_url,
                },
            }
        )
        await self._redis.rpush(RESEARCH_QUEUE_KEY, job)
        await self._redis.set(redis_key, "1", ex=60 * 60 * 24 * 14)
        stats.events_enqueued += 1

    async def _flush_upsert_buffer(self, stats: IngestStats | None = None) -> None:
        if not self._upsert_buffer:
            return
        points = list(self._upsert_buffer)
        self._upsert_buffer.clear()
        texts = [p.payload["text"] for p in points]
        vectors = await self._embed_all(texts)
        for point, vector in zip(points, vectors, strict=True):
            point.vector = vector
        if not self.dry_run:
            await self._qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
            logger.debug("Upserted %d points.", len(points))
            if self.upsert_delay > 0:
                await asyncio.sleep(self.upsert_delay)
        if stats is not None:
            stats.points_upserted += len(points)

    async def _embed_all(self, texts: list[str]) -> list[list[float]]:
        batches = [texts[i : i + self.embed_batch_size] for i in range(0, len(texts), self.embed_batch_size)]
        results: list[list[float]] = []
        for group_start in range(0, len(batches), self.embed_concurrency):
            group = batches[group_start : group_start + self.embed_concurrency]
            group_results = await asyncio.gather(*[self._embed_batch(b) for b in group])
            for batch_vecs in group_results:
                results.extend(batch_vecs)
        return results

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        async with self._embed_semaphore:
            for attempt in range(4):
                try:
                    async with httpx.AsyncClient(timeout=45.0) as http:
                        resp = await http.post(
                            HF_EMBEDDING_URL,
                            headers={"Authorization": f"Bearer {HF_TOKEN}"},
                            json={"inputs": texts, "options": {"wait_for_model": True}},
                        )
                        if resp.status_code == 429:
                            wait = 30 * (attempt + 1)
                            logger.warning("HF embedding 429 — waiting %ds", wait)
                            await asyncio.sleep(wait)
                            continue
                        resp.raise_for_status()
                        result = resp.json()
                        if isinstance(result, list) and result and isinstance(result[0], list):
                            return result
                        if isinstance(result, list) and result and isinstance(result[0], (int, float)):
                            return [result]
                        break
                except Exception as exc:
                    logger.warning("Embed attempt %d failed: %s", attempt + 1, exc)
                    await asyncio.sleep(5 * (attempt + 1))
        return [[0.0] * EMBEDDING_DIM for _ in texts]

    async def _save_checkpoint(self, ts_str: str) -> None:
        if self.dry_run or not self._redis:
            return
        await self._redis.set(CHECKPOINT_KEY, ts_str)

    async def _ensure_collection(self) -> None:
        if self.dry_run:
            logger.info("[dry-run] Skipping collection creation.")
            return
        exists = await self._qdrant.collection_exists(COLLECTION_NAME)
        if not exists:
            await self._qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=qdrant_models.VectorParams(
                    size=EMBEDDING_DIM,
                    distance=qdrant_models.Distance.COSINE,
                ),
                optimizers_config=qdrant_models.OptimizersConfigDiff(indexing_threshold=1000),
            )
            logger.info("Created Qdrant collection '%s'.", COLLECTION_NAME)
        else:
            info = await self._qdrant.get_collection(COLLECTION_NAME)
            logger.info("Collection '%s' ready (%d points).", COLLECTION_NAME, info.points_count or 0)

        # Ensure payload indexes exist for the fields used in filtered searches
        # and score-boost queries.  create_payload_index is idempotent — safe to
        # call on every run.
        keyword_fields = [
            "quad_class",
            "event_root_code",
            "cameo_description",
            "actor1_name",
            "actor1_country",
            "actor1_country_name",
            "actor1_type",
            "actor2_name",
            "actor2_country",
            "actor2_country_name",
            "actor2_type",
            "country",
            "country_name",
            "action_geo",
            "pub_date",
            "provenance",
        ]
        float_fields = [
            "num_mentions",
            "avg_tone",
            "goldstein_scale",
            "mentions_weight",
            "ingested_at_unix",
        ]
        for field_name in keyword_fields:
            await self._qdrant.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
            )
        for field_name in float_fields:
            await self._qdrant.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=qdrant_models.PayloadSchemaType.FLOAT,
            )
        logger.info("Payload indexes verified for '%s'.", COLLECTION_NAME)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ingest GDELT 2.0 global events into OSIA Qdrant knowledge base",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dry-run", action="store_true", help="Skip Qdrant writes and Redis updates")
    p.add_argument("--resume", action="store_true", help="Resume from last Redis checkpoint")
    p.add_argument("--days-back", type=int, default=7, help="Days to backfill on first run")
    p.add_argument("--min-mentions", type=int, default=10, help="Minimum NumMentions for conflict events")
    p.add_argument("--enqueue-notable", action="store_true", help="Push high-coverage events to research queue")
    p.add_argument("--limit", type=int, default=0, help="Stop after N events ingested (0=no limit)")
    p.add_argument("--embed-batch-size", type=int, default=48, help="Texts per HF embedding call")
    p.add_argument("--embed-concurrency", type=int, default=3, help="Parallel embedding calls")
    p.add_argument("--upsert-batch-size", type=int, default=32, help="Points per Qdrant upsert call")
    p.add_argument("--upsert-delay", type=float, default=0.5, help="Seconds to sleep after each Qdrant upsert call")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    ingestor = GdeltIngestor(args)
    asyncio.run(ingestor.run())

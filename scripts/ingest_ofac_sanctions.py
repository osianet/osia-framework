"""
OSIA OFAC SDN Sanctions List Ingestion

Downloads the OFAC Specially Designated Nationals (SDN) list XML from the US
Treasury, parses every entry (Individual, Entity, Vessel, Aircraft), builds
rich structured documents, and upserts into the 'ofac-sanctions' Qdrant
collection for Finance and Geopolitical desk RAG retrieval.

The SDN list is updated frequently (check Publish_Date in XML header).
Re-running this script will overwrite existing points by stable UID-based IDs
so it is safe to run as a periodic refresh.

Usage:
  uv run python scripts/ingest_ofac_sanctions.py
  uv run python scripts/ingest_ofac_sanctions.py --dry-run
  uv run python scripts/ingest_ofac_sanctions.py --program-filter RUSSIA IRAN
  uv run python scripts/ingest_ofac_sanctions.py --enqueue-notable
  uv run python scripts/ingest_ofac_sanctions.py --sdn-type Individual

Options:
  --dry-run             Parse but skip Qdrant writes and Redis enqueuing
  --program-filter      Only ingest entries from these sanctions programs
                        e.g. RUSSIA IRAN CUBA DPRK SDGT
  --sdn-type            Only ingest these entity types: Individual Entity Vessel Aircraft
  --enqueue-notable     Push high-interest entries to desk research queues:
                        financial programs → Finance desk
                        geopolitical programs → Geopolitical desk
  --embed-batch-size    Texts per HF embedding call (default: 64)
  --embed-concurrency   Parallel embedding calls (default: 4)
  --upsert-batch-size   Points per Qdrant upsert call (default: 100)

Environment variables (from .env):
  HF_TOKEN              HuggingFace token (required for embeddings)
  QDRANT_URL            Qdrant URL (default: https://qdrant.osia.dev)
  QDRANT_API_KEY        Qdrant API key
  REDIS_URL             Redis URL (default: redis://localhost:6379)
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import UTC, datetime
from io import BytesIO

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
logger = logging.getLogger("osia.ofac_ingest")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN", "")
QDRANT_URL = os.getenv("QDRANT_URL", "https://qdrant.osia.dev")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "") or None
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

COLLECTION_NAME = "ofac-sanctions"
EMBEDDING_DIM = 384
SOURCE_LABEL = "OFAC Specially Designated Nationals (SDN) List"

SDN_XML_URL = "https://www.treasury.gov/ofac/downloads/sdn.xml"

HF_EMBEDDING_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
)

# XML namespace used in the SDN document
_NS = "https://sanctionslistservice.ofac.treas.gov/api/PublicationPreview/exports/XML"

RESEARCH_QUEUE_KEY = "osia:research_queue"
TODAY = datetime.now(UTC).strftime("%Y-%m-%d")

# Programs to route to the Finance desk research queue (corruption, dark money, drugs, fraud)
FINANCE_PROGRAMS = {
    "GLOMAG",  # Global Magnitsky — corruption and human rights abusers
    "ILLICIT-DRUGS-EO14059",
    "SDNTK",  # Narcotics Kingpins
    "TCO",  # Transnational Criminal Organisations
    "CAATSA - RUSSIA",  # Finance-adjacent Russia sanctions
    "IFSR",  # Iran Financial Sanctions Regulations
    "Nicaragua-EO13851",
    "VENEZUELA-EO13850",
}

# Programs to route to the Geopolitical desk research queue
GEO_PROGRAMS = {
    "RUSSIA-EO14024",
    "UKRAINE-EO13661",
    "UKRAINE-EO13662",
    "UKRAINE-EO13685",
    "IRAN",
    "IRAN-EO13876",
    "IRAN-HR",
    "DPRK",
    "DPRK2",
    "DPRK3",
    "DPRK4",
    "SYRIA",
    "SDGT",  # Specially Designated Global Terrorists
    "CUBA",
    "BELARUS-EO14038",
    "MYANMAR-EO14014",
    "SUDAN",
    "SOMALIA",
    "MALI",
    "CAR",  # Central African Republic
    "LIBYA2",
    "YEMEN",
}


# ---------------------------------------------------------------------------
# XML parsing helpers
# ---------------------------------------------------------------------------


def _tag(local: str) -> str:
    """Return the fully-qualified tag name with SDN namespace."""
    return f"{{{_NS}}}{local}"


def _text(element: ET.Element | None, default: str = "") -> str:
    if element is None or element.text is None:
        return default
    return element.text.strip()


def parse_entry(entry: ET.Element) -> dict:
    """
    Extract all fields from a single <sdnEntry> element into a flat dict
    with lists for multi-value fields.
    """

    def find(tag: str) -> ET.Element | None:
        return entry.find(_tag(tag))

    def findall(path: str) -> list[ET.Element]:
        parts = path.split("/")
        elements = [entry]
        for part in parts:
            next_elements = []
            for el in elements:
                next_elements.extend(el.findall(_tag(part)))
            elements = next_elements
        return elements

    uid = _text(find("uid"))
    first_name = _text(find("firstName"))
    last_name = _text(find("lastName"))
    sdn_type = _text(find("sdnType"))
    title = _text(find("title"))
    remarks = _text(find("remarks"))
    call_sign = _text(find("callSign"))
    vessel_type = _text(find("vesselType"))
    vessel_flag = _text(find("vesselFlag"))
    vessel_owner = _text(find("vesselOwner"))
    tonnage = _text(find("tonnage"))
    grt = _text(find("GRT"))

    # Sanctions programs
    programs = [_text(p) for p in findall("programList/program") if _text(p)]

    # Aliases
    akas: list[dict] = []
    for aka in findall("akaList/aka"):
        aka_type = _text(aka.find(_tag("type")))
        aka_cat = _text(aka.find(_tag("category")))
        aka_first = _text(aka.find(_tag("firstName")))
        aka_last = _text(aka.find(_tag("lastName")))
        name_parts = [p for p in [aka_first, aka_last] if p]
        if name_parts:
            akas.append({"name": " ".join(name_parts), "type": aka_type, "category": aka_cat})

    # Addresses
    addresses: list[str] = []
    for addr in findall("addressList/address"):
        parts = []
        for field_name in ("address1", "address2", "address3", "city", "stateOrProvince", "postalCode", "country"):
            val = _text(addr.find(_tag(field_name)))
            if val:
                parts.append(val)
        if parts:
            addresses.append(", ".join(parts))

    # Identification documents
    ids: list[dict] = []
    for id_el in findall("idList/id"):
        id_type = _text(id_el.find(_tag("idType")))
        id_number = _text(id_el.find(_tag("idNumber")))
        id_country = _text(id_el.find(_tag("idCountry")))
        if id_type or id_number:
            ids.append({"type": id_type, "number": id_number, "country": id_country})

    # Dates of birth
    dobs = [
        _text(d.find(_tag("dateOfBirth")))
        for d in findall("dateOfBirthList/dateOfBirthItem")
        if _text(d.find(_tag("dateOfBirth")))
    ]

    # Places of birth
    pobs = [
        _text(p.find(_tag("placeOfBirth")))
        for p in findall("placeOfBirthList/placeOfBirthItem")
        if _text(p.find(_tag("placeOfBirth")))
    ]

    # Nationalities
    nationalities = [
        _text(n.find(_tag("nationality")))
        for n in findall("nationalityList/nationality")
        if _text(n.find(_tag("nationality")))
    ]

    # Citizenship
    citizenships = [
        _text(c.find(_tag("country"))) for c in findall("citizenshipList/citizenship") if _text(c.find(_tag("country")))
    ]

    return {
        "uid": uid,
        "first_name": first_name,
        "last_name": last_name,
        "sdn_type": sdn_type,
        "title": title,
        "remarks": remarks,
        "programs": programs,
        "akas": akas,
        "addresses": addresses,
        "ids": ids,
        "dobs": dobs,
        "pobs": pobs,
        "nationalities": nationalities,
        "citizenships": citizenships,
        "call_sign": call_sign,
        "vessel_type": vessel_type,
        "vessel_flag": vessel_flag,
        "vessel_owner": vessel_owner,
        "tonnage": tonnage,
        "grt": grt,
    }


# ---------------------------------------------------------------------------
# Document builder
# ---------------------------------------------------------------------------


def build_document(e: dict) -> str:
    """Build a rich structured text document from a parsed SDN entry."""
    lines: list[str] = []

    # Full name
    name_parts = [p for p in [e["first_name"], e["last_name"]] if p]
    full_name = " ".join(name_parts)

    # Lead with a prose narrative sentence for better semantic embedding
    if full_name or e["sdn_type"]:
        entity_desc = full_name if full_name else "an unnamed entity"
        type_label = e["sdn_type"].lower() if e["sdn_type"] else "entity"
        prog_str = f" under {', '.join(e['programs'][:3])}" if e["programs"] else ""
        nat_str = f" of {e['nationalities'][0]} nationality" if e["nationalities"] else ""
        lines.append(f"{entity_desc} is a US Treasury OFAC-sanctioned {type_label}{nat_str}{prog_str}.")

    lines.append(f"OFAC Sanctions Entry — {e['sdn_type']}")
    if full_name:
        lines.append(f"Name: {full_name}")
    if e["title"]:
        lines.append(f"Title / Role: {e['title']}")
    if e["programs"]:
        lines.append(f"Sanctions Programs: {', '.join(e['programs'])}")

    # Aliases
    if e["akas"]:
        strong = [a["name"] for a in e["akas"] if a["category"] == "strong"]
        weak = [a["name"] for a in e["akas"] if a["category"] != "strong"]
        if strong:
            lines.append(f"Also Known As (strong): {'; '.join(strong)}")
        if weak:
            lines.append(f"Also Known As (weak): {'; '.join(weak)}")

    # Individual-specific fields
    if e["dobs"]:
        lines.append(f"Date of Birth: {'; '.join(e['dobs'])}")
    if e["pobs"]:
        lines.append(f"Place of Birth: {'; '.join(e['pobs'])}")
    if e["nationalities"]:
        lines.append(f"Nationality: {'; '.join(e['nationalities'])}")
    if e["citizenships"]:
        lines.append(f"Citizenship: {'; '.join(e['citizenships'])}")

    # Addresses
    if e["addresses"]:
        lines.append("")
        lines.append("Known Addresses:")
        for addr in e["addresses"]:
            lines.append(f"  - {addr}")

    # Identification documents
    if e["ids"]:
        lines.append("")
        lines.append("Identification Documents:")
        for doc in e["ids"]:
            id_str = doc["type"]
            if doc["number"]:
                id_str += f": {doc['number']}"
            if doc["country"]:
                id_str += f" ({doc['country']})"
            lines.append(f"  - {id_str}")

    # Vessel-specific fields
    if e["sdn_type"] == "Vessel":
        if e["call_sign"]:
            lines.append(f"Call Sign: {e['call_sign']}")
        if e["vessel_type"]:
            lines.append(f"Vessel Type: {e['vessel_type']}")
        if e["vessel_flag"]:
            lines.append(f"Flag: {e['vessel_flag']}")
        if e["vessel_owner"]:
            lines.append(f"Owner: {e['vessel_owner']}")
        if e["tonnage"]:
            lines.append(f"Tonnage: {e['tonnage']}")
        if e["grt"]:
            lines.append(f"GRT: {e['grt']}")

    # Remarks
    if e["remarks"]:
        lines.append("")
        lines.append(f"Remarks: {e['remarks']}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class IngestStats:
    records_seen: int = 0
    records_skipped: int = 0
    records_processed: int = 0
    points_upserted: int = 0
    finance_enqueued: int = 0
    geo_enqueued: int = 0
    errors: int = 0
    publish_date: str = ""
    started_at: float = field(default_factory=time.time)

    def elapsed(self) -> str:
        secs = int(time.time() - self.started_at)
        return f"{secs // 60}m{secs % 60:02d}s"

    def log_progress(self) -> None:
        logger.info(
            "seen=%d processed=%d skipped=%d upserted=%d finance_q=%d geo_q=%d errors=%d elapsed=%s",
            self.records_seen,
            self.records_processed,
            self.records_skipped,
            self.points_upserted,
            self.finance_enqueued,
            self.geo_enqueued,
            self.errors,
            self.elapsed(),
        )


# ---------------------------------------------------------------------------
# Main ingestor
# ---------------------------------------------------------------------------


class OfacSdnIngestor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.dry_run: bool = args.dry_run
        self.enqueue_notable: bool = args.enqueue_notable
        self.program_filter: set[str] | None = {p.upper() for p in args.program_filter} if args.program_filter else None
        self.sdn_type_filter: set[str] | None = {t.capitalize() for t in args.sdn_type} if args.sdn_type else None
        self.embed_batch_size: int = args.embed_batch_size
        self.embed_concurrency: int = args.embed_concurrency
        self.upsert_batch_size: int = args.upsert_batch_size

        self._qdrant = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, port=None)
        self._redis: aioredis.Redis | None = None
        self._embed_semaphore = asyncio.Semaphore(self.embed_concurrency)
        self._upsert_buffer: list[qdrant_models.PointStruct] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self._redis = aioredis.from_url(REDIS_URL, decode_responses=True)
        try:
            await self._ensure_collection()

            logger.info("Downloading SDN XML from %s ...", SDN_XML_URL)
            xml_bytes = await asyncio.get_event_loop().run_in_executor(None, self._download_xml)
            logger.info("Downloaded %.1f MB", len(xml_bytes) / (1024 * 1024))

            stats = IngestStats()
            await self._ingest(xml_bytes, stats)

            await self._flush_upsert_buffer(stats)
            stats.log_progress()
            logger.info(
                "Ingestion complete. Publish date: %s | Total records in list: see log above",
                stats.publish_date,
            )
        finally:
            await self._qdrant.close()
            if self._redis:
                await self._redis.aclose()

    def _download_xml(self) -> bytes:
        with httpx.Client(timeout=120.0, follow_redirects=True) as http:
            resp = http.get(SDN_XML_URL)
            resp.raise_for_status()
            return resp.content

    # ------------------------------------------------------------------
    # Collection setup
    # ------------------------------------------------------------------

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
                optimizers_config=qdrant_models.OptimizersConfigDiff(
                    indexing_threshold=1000,
                ),
            )
            logger.info("Created Qdrant collection '%s'.", COLLECTION_NAME)
        else:
            info = await self._qdrant.get_collection(COLLECTION_NAME)
            logger.info(
                "Collection '%s' ready (%d points).",
                COLLECTION_NAME,
                info.points_count or 0,
            )

        # Ensure payload indexes exist (idempotent — safe to call on every run).
        keyword_fields = ["sdn_type", "entity_name", "document_type", "provenance"]
        float_fields = ["ingested_at_unix"]
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

    # ------------------------------------------------------------------
    # XML ingestion
    # ------------------------------------------------------------------

    async def _ingest(self, xml_bytes: bytes, stats: IngestStats) -> None:
        loop = asyncio.get_event_loop()

        def _parse_all() -> list[dict]:
            entries = []
            publish_date = ""
            for _event, elem in ET.iterparse(BytesIO(xml_bytes), events=("end",)):
                if elem.tag == _tag("Publish_Date"):
                    publish_date = elem.text or ""
                    logger.info("SDN list publish date: %s", publish_date)
                elif elem.tag == _tag("Record_Count"):
                    logger.info("SDN list record count: %s", elem.text)
                elif elem.tag == _tag("sdnEntry"):
                    entries.append(parse_entry(elem))
                    elem.clear()  # free memory
            return entries, publish_date

        logger.info("Parsing XML (iterparse)...")
        all_entries, publish_date = await loop.run_in_executor(None, _parse_all)
        stats.publish_date = publish_date
        logger.info("Parsed %d entries.", len(all_entries))

        for entry in all_entries:
            stats.records_seen += 1
            try:
                await self._process_entry(entry, stats)
            except Exception as exc:
                stats.errors += 1
                logger.warning("Error processing entry uid=%s: %s", entry.get("uid"), exc)

            if stats.records_seen % 2000 == 0:
                stats.log_progress()

    async def _process_entry(self, e: dict, stats: IngestStats) -> None:
        # Type filter
        if self.sdn_type_filter and e["sdn_type"] not in self.sdn_type_filter:
            stats.records_skipped += 1
            return

        # Program filter
        programs_upper = {p.upper() for p in e["programs"]}
        if self.program_filter and not programs_upper & self.program_filter:
            stats.records_skipped += 1
            return

        doc = build_document(e)
        if not doc.strip():
            stats.records_skipped += 1
            return

        stats.records_processed += 1

        # Stable point ID from SDN uid — safe to re-run as an upsert refresh
        point_id = str(uuid.UUID(bytes=hashlib.sha256(f"ofac:{e['uid']}".encode()).digest()[:16]))

        name_parts = [p for p in [e["first_name"], e["last_name"]] if p]
        full_name = " ".join(name_parts)
        aka_names = [a["name"] for a in e["akas"]]

        entity_tags = [t for t in [full_name, *aka_names[:5], *e["programs"][:3]] if t]

        payload: dict = {
            "text": doc,
            "source": SOURCE_LABEL,
            "document_type": "sanctions_entry",
            "provenance": "ofac_sdn_list",
            "sdn_uid": e["uid"],
            "sdn_type": e["sdn_type"],
            "ingest_date": TODAY,
            "publish_date": stats.publish_date,
            "ingested_at_unix": int(time.time()),
            "entity_tags": entity_tags,
        }
        if full_name:
            payload["entity_name"] = full_name
        if e["programs"]:
            payload["programs"] = e["programs"]
        if e["nationalities"]:
            payload["nationalities"] = e["nationalities"]
        if e["dobs"]:
            payload["date_of_birth"] = e["dobs"][0]
        if e["addresses"]:
            payload["addresses"] = e["addresses"][:5]

        self._upsert_buffer.append(
            qdrant_models.PointStruct(
                id=point_id,
                vector=[0.0] * EMBEDDING_DIM,
                payload=payload,
            )
        )

        if len(self._upsert_buffer) >= self.upsert_batch_size:
            await self._flush_upsert_buffer(stats)

        # Research queue enqueuing
        if self.enqueue_notable:
            await self._maybe_enqueue(e, full_name, programs_upper, stats)

    # ------------------------------------------------------------------
    # Research queue
    # ------------------------------------------------------------------

    async def _maybe_enqueue(
        self,
        e: dict,
        full_name: str,
        programs_upper: set[str],
        stats: IngestStats,
    ) -> None:
        if not self._redis or self.dry_run or not full_name:
            return

        # Only enqueue Individuals and named Entities — skip vessels/aircraft
        if e["sdn_type"] not in ("Individual", "Entity"):
            return

        redis_key = f"osia:ofac:enqueued:{e['uid']}"
        if await self._redis.exists(redis_key):
            return

        # Route to Finance or Geopolitical desk based on program
        desk = None
        if programs_upper & {p.upper() for p in FINANCE_PROGRAMS}:
            desk = "finance-and-economics-directorate"
        elif programs_upper & {p.upper() for p in GEO_PROGRAMS}:
            desk = "geopolitical-and-security-desk"

        if not desk:
            return

        program_str = ", ".join(e["programs"][:3])
        job = json.dumps(
            {
                "job_id": str(uuid.uuid4()),
                "topic": f"OFAC sanctioned {e['sdn_type'].lower()}: {full_name} (programs: {program_str})",
                "desk": desk,
                "priority": "low",
                "triggered_by": "ofac_sdn_ingest",
                "metadata": {
                    "sdn_uid": e["uid"],
                    "sdn_type": e["sdn_type"],
                    "programs": e["programs"],
                    "entity_name": full_name,
                },
            }
        )
        await self._redis.rpush(RESEARCH_QUEUE_KEY, job)
        await self._redis.set(redis_key, "1", ex=60 * 60 * 24 * 30)

        if desk == "finance-and-economics-directorate":
            stats.finance_enqueued += 1
        else:
            stats.geo_enqueued += 1

    # ------------------------------------------------------------------
    # Flush + embed
    # ------------------------------------------------------------------

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
            logger.debug("Upserted %d points to '%s'.", len(points), COLLECTION_NAME)

        if stats is not None:
            stats.points_upserted += len(points)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

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
                        logger.error("Unexpected HF embedding shape: %s", type(result))
                        break
                except Exception as exc:
                    logger.warning("Embed attempt %d failed: %s", attempt + 1, exc)
                    await asyncio.sleep(5 * (attempt + 1))
            logger.error("Embedding failed for batch of %d — using zero vectors.", len(texts))
            return [[0.0] * EMBEDDING_DIM for _ in texts]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ingest OFAC SDN sanctions list into OSIA Qdrant knowledge base",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Parse but skip Qdrant writes and Redis enqueuing",
    )
    p.add_argument(
        "--program-filter",
        nargs="+",
        metavar="PROGRAM",
        dest="program_filter",
        help="Only ingest entries under these sanctions programs (e.g. RUSSIA IRAN SDGT)",
    )
    p.add_argument(
        "--sdn-type",
        nargs="+",
        metavar="TYPE",
        dest="sdn_type",
        choices=["Individual", "Entity", "Vessel", "Aircraft"],
        help="Only ingest these SDN types",
    )
    p.add_argument(
        "--enqueue-notable",
        action="store_true",
        dest="enqueue_notable",
        help="Push high-interest entries to Finance/Geopolitical desk research queues",
    )
    p.add_argument("--embed-batch-size", type=int, default=64, dest="embed_batch_size")
    p.add_argument("--embed-concurrency", type=int, default=4, dest="embed_concurrency")
    p.add_argument("--upsert-batch-size", type=int, default=100, dest="upsert_batch_size")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not HF_TOKEN:
        parser.error("HF_TOKEN not set in environment — required for embeddings.")

    logger.info(
        "OFAC SDN ingest | program_filter=%s sdn_type=%s enqueue_notable=%s dry_run=%s",
        args.program_filter or "all",
        args.sdn_type or "all",
        args.enqueue_notable,
        args.dry_run,
    )

    if args.dry_run:
        logger.warning("DRY RUN — no data will be written to Qdrant or Redis.")

    ingestor = OfacSdnIngestor(args)
    asyncio.run(ingestor.run())


if __name__ == "__main__":
    main()

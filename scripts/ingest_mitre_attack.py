"""
OSIA MITRE ATT&CK Ingestion

Downloads the MITRE ATT&CK STIX 2.1 knowledge base from GitHub (enterprise,
and optionally mobile and ICS domains), parses the object graph, and upserts
rich documents for each entity type — techniques, threat actor groups, software
(malware/tools), and mitigations — into a dedicated 'mitre-attack' Qdrant
collection.

Relationship data (group→technique, group→software, software→technique) is
resolved before writing so every group document includes its attributed
techniques and malware, and every technique document lists known groups and
software that use it.

Source:
  https://github.com/mitre-attack/attack-stix-data
  enterprise-attack/enterprise-attack.json  (Apache 2.0)
  mobile-attack/mobile-attack.json          (Apache 2.0)
  ics-attack/ics-attack.json                (Apache 2.0)

Usage:
  uv run python scripts/ingest_mitre_attack.py
  uv run python scripts/ingest_mitre_attack.py --domains enterprise mobile ics
  uv run python scripts/ingest_mitre_attack.py --local-dir /tmp/attack-stix
  uv run python scripts/ingest_mitre_attack.py --dry-run

Options:
  --domains         ATT&CK domains to ingest (default: enterprise)
  --local-dir       Path to a local directory containing <domain>-attack.json files.
                    If set, skips the GitHub download.
  --dry-run         Parse and build documents but skip Qdrant writes
  --embed-batch-size  Texts per HF embedding call (default: 48)
  --embed-concurrency Parallel embedding calls (default: 3)
  --upsert-batch-size Points per Qdrant upsert call (default: 64)

Environment variables (from .env):
  HF_TOKEN          HuggingFace token (required — for embeddings)
  QDRANT_URL        Qdrant URL (default: https://qdrant.osia.dev)
  QDRANT_API_KEY    Qdrant API key
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("osia.mitre_attack_ingest")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN", "")
QDRANT_URL = os.getenv("QDRANT_URL", "https://qdrant.osia.dev")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "") or None

COLLECTION_NAME = "mitre-attack"
EMBEDDING_DIM = 384

GITHUB_RAW = "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master"
DOMAIN_URLS = {
    "enterprise": f"{GITHUB_RAW}/enterprise-attack/enterprise-attack.json",
    "mobile": f"{GITHUB_RAW}/mobile-attack/mobile-attack.json",
    "ics": f"{GITHUB_RAW}/ics-attack/ics-attack.json",
}

HF_EMBEDDING_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
)

TODAY = datetime.now(UTC).strftime("%Y-%m-%d")

# Object types we care about
TECHNIQUE_TYPE = "attack-pattern"
GROUP_TYPE = "intrusion-set"
SOFTWARE_TYPES = {"malware", "tool"}
MITIGATION_TYPE = "course-of-action"
RELATIONSHIP_TYPE = "relationship"


# ---------------------------------------------------------------------------
# STIX helpers
# ---------------------------------------------------------------------------


def _mitre_id(obj: dict) -> str:
    """Extract the MITRE ATT&CK external ID (T1059, G0007, S0023, M1017)."""
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return ref.get("external_id", "")
    return ""


def _mitre_url(obj: dict) -> str:
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return ref.get("url", "")
    return ""


def _is_active(obj: dict) -> bool:
    return not (obj.get("x-mitre-deprecated") or obj.get("revoked"))


def _tactics(obj: dict) -> list[str]:
    return [p.get("phase_name", "") for p in obj.get("kill_chain_phases", []) if p.get("phase_name")]


def _platforms(obj: dict) -> list[str]:
    return [str(p) for p in obj.get("x-mitre-platforms", [])]


def _safe(val, default: str = "") -> str:
    if val is None:
        return default
    s = str(val).strip()
    return s if s not in ("None", "nan", "") else default


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


class AttackGraph:
    """Parses a STIX bundle and builds lookup tables used by document builders."""

    def __init__(self) -> None:
        self.objects: dict[str, dict] = {}  # stix_id → object
        self.group_uses_techniques: dict[str, list[tuple[str, str]]] = {}  # group_id → [(t_id, t_name)]
        self.group_uses_software: dict[str, list[tuple[str, str]]] = {}  # group_id → [(s_id, s_name)]
        self.technique_used_by: dict[str, list[tuple[str, str]]] = {}  # tech_id → [(g_id, g_name)]
        self.software_used_by: dict[str, list[tuple[str, str]]] = {}  # soft_id → [(g_id, g_name)]
        self.software_uses_techniques: dict[str, list[tuple[str, str]]] = {}  # soft_id → [(t_id, t_name)]

    def ingest_bundle(self, bundle: dict, domain: str) -> None:
        objects = bundle.get("objects", [])
        relationships: list[dict] = []

        for obj in objects:
            obj_type = obj.get("type", "")
            obj_id = obj.get("id", "")
            if not obj_id:
                continue
            # Store all objects (even deprecated — needed to resolve relationship refs)
            self.objects[obj_id] = {**obj, "_domain": domain}
            if obj_type == RELATIONSHIP_TYPE:
                relationships.append(obj)

        # Build relationship maps
        for rel in relationships:
            if not _is_active(rel):
                continue
            rel_type = rel.get("relationship_type", "")
            src = rel.get("source_ref", "")
            tgt = rel.get("target_ref", "")
            src_obj = self.objects.get(src, {})
            tgt_obj = self.objects.get(tgt, {})
            if not src_obj or not tgt_obj:
                continue

            src_type = src_obj.get("type", "")
            tgt_type = tgt_obj.get("type", "")

            if rel_type == "uses":
                if src_type == GROUP_TYPE and tgt_type == TECHNIQUE_TYPE:
                    t_name = _safe(tgt_obj.get("name"))
                    t_id = _mitre_id(tgt_obj)
                    if t_name or t_id:
                        self.group_uses_techniques.setdefault(src, []).append((t_id, t_name))
                        self.technique_used_by.setdefault(tgt, []).append(
                            (_mitre_id(src_obj), _safe(src_obj.get("name")))
                        )
                elif src_type == GROUP_TYPE and tgt_type in SOFTWARE_TYPES:
                    s_name = _safe(tgt_obj.get("name"))
                    s_id = _mitre_id(tgt_obj)
                    if s_name or s_id:
                        self.group_uses_software.setdefault(src, []).append((s_id, s_name))
                        self.software_used_by.setdefault(tgt, []).append(
                            (_mitre_id(src_obj), _safe(src_obj.get("name")))
                        )
                elif tgt_type in SOFTWARE_TYPES and src_type == TECHNIQUE_TYPE:
                    # reverse: technique used by software? — usually software uses technique
                    pass
                elif src_type in SOFTWARE_TYPES and tgt_type == TECHNIQUE_TYPE:
                    t_name = _safe(tgt_obj.get("name"))
                    t_id = _mitre_id(tgt_obj)
                    if t_name or t_id:
                        self.software_uses_techniques.setdefault(src, []).append((t_id, t_name))

    def active_objects_of_type(self, *types: str):
        return [obj for obj in self.objects.values() if obj.get("type") in types and _is_active(obj)]


# ---------------------------------------------------------------------------
# Document builders
# ---------------------------------------------------------------------------


def build_technique_doc(obj: dict, graph: AttackGraph) -> str:
    attack_id = _mitre_id(obj)
    name = _safe(obj.get("name"))
    desc = _safe(obj.get("description"))
    tactics = _tactics(obj)
    platforms = _platforms(obj)
    detection = _safe(obj.get("x-mitre-detection"))
    data_sources = [str(s) for s in obj.get("x-mitre-data-sources", [])]
    is_sub = obj.get("x-mitre-is-subtechnique", False)
    domain = obj.get("_domain", "enterprise")

    header = f"ATT&CK {'Sub-Technique' if is_sub else 'Technique'}: {attack_id} — {name}"
    parts = [header]

    meta_lines = []
    if tactics:
        meta_lines.append(f"Tactic: {', '.join(t.replace('-', ' ').title() for t in tactics)}")
    if platforms:
        meta_lines.append(f"Platforms: {', '.join(platforms)}")
    if domain != "enterprise":
        meta_lines.append(f"Domain: {domain.title()}")
    if meta_lines:
        parts.append("\n".join(meta_lines))

    if desc:
        parts.append(desc[:3000])  # cap to avoid over-long embeddings

    if detection:
        parts.append(f"Detection:\n{detection[:1000]}")

    if data_sources:
        parts.append(f"Data Sources: {', '.join(data_sources[:10])}")

    obj_id = obj.get("id", "")
    used_by = graph.technique_used_by.get(obj_id, [])
    if used_by:
        group_labels = [f"{gid} ({gname})" if gid else gname for gid, gname in used_by[:20]]
        parts.append(f"Known Groups: {', '.join(group_labels)}")

    return "\n\n".join(parts)


def build_group_doc(obj: dict, graph: AttackGraph) -> str:
    attack_id = _mitre_id(obj)
    name = _safe(obj.get("name"))
    desc = _safe(obj.get("description"))
    aliases = [str(a) for a in obj.get("aliases", []) if str(a) != name]

    header = f"ATT&CK Group: {attack_id} — {name}"
    parts = [header]

    meta_lines = []
    if aliases:
        meta_lines.append(f"Also Known As: {', '.join(aliases)}")
    if meta_lines:
        parts.append("\n".join(meta_lines))

    if desc:
        parts.append(desc[:3000])

    obj_id = obj.get("id", "")

    techniques = graph.group_uses_techniques.get(obj_id, [])
    if techniques:
        tech_labels = [f"{tid} ({tname})" if tid else tname for tid, tname in techniques[:30]]
        parts.append(f"Techniques Used: {', '.join(tech_labels)}")

    software = graph.group_uses_software.get(obj_id, [])
    if software:
        soft_labels = [f"{sid} ({sname})" if sid else sname for sid, sname in software[:20]]
        parts.append(f"Malware/Tools: {', '.join(soft_labels)}")

    return "\n\n".join(parts)


def build_software_doc(obj: dict, graph: AttackGraph) -> str:
    attack_id = _mitre_id(obj)
    name = _safe(obj.get("name"))
    desc = _safe(obj.get("description"))
    soft_type = obj.get("type", "malware").title()
    platforms = _platforms(obj)
    aliases = [str(a) for a in obj.get("x-mitre-aliases", []) if str(a) != name]

    header = f"ATT&CK Software: {attack_id} — {name} ({soft_type})"
    parts = [header]

    meta_lines = []
    if aliases:
        meta_lines.append(f"Also Known As: {', '.join(aliases)}")
    if platforms:
        meta_lines.append(f"Platforms: {', '.join(platforms)}")
    if meta_lines:
        parts.append("\n".join(meta_lines))

    if desc:
        parts.append(desc[:3000])

    obj_id = obj.get("id", "")

    used_by = graph.software_used_by.get(obj_id, [])
    if used_by:
        group_labels = [f"{gid} ({gname})" if gid else gname for gid, gname in used_by[:15]]
        parts.append(f"Used By: {', '.join(group_labels)}")

    techniques = graph.software_uses_techniques.get(obj_id, [])
    if techniques:
        tech_labels = [f"{tid} ({tname})" if tid else tname for tid, tname in techniques[:20]]
        parts.append(f"Techniques Implemented: {', '.join(tech_labels)}")

    return "\n\n".join(parts)


def build_mitigation_doc(obj: dict) -> str:
    attack_id = _mitre_id(obj)
    name = _safe(obj.get("name"))
    desc = _safe(obj.get("description"))

    header = f"ATT&CK Mitigation: {attack_id} — {name}"
    parts = [header]
    if desc:
        parts.append(desc[:2000])
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class IngestStats:
    techniques: int = 0
    groups: int = 0
    software: int = 0
    mitigations: int = 0
    points_upserted: int = 0
    errors: int = 0

    def total(self) -> int:
        return self.techniques + self.groups + self.software + self.mitigations

    def log_summary(self) -> None:
        logger.info(
            "techniques=%d groups=%d software=%d mitigations=%d upserted=%d errors=%d",
            self.techniques,
            self.groups,
            self.software,
            self.mitigations,
            self.points_upserted,
            self.errors,
        )


# ---------------------------------------------------------------------------
# Main ingestor
# ---------------------------------------------------------------------------


class MitreAttackIngestor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.dry_run: bool = args.dry_run
        self.domains: list[str] = args.domains
        self.local_dir: Path | None = Path(args.local_dir) if args.local_dir else None
        self.embed_batch_size: int = args.embed_batch_size
        self.embed_concurrency: int = args.embed_concurrency
        self.upsert_batch_size: int = args.upsert_batch_size

        self._qdrant = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, port=None)
        self._embed_semaphore = asyncio.Semaphore(self.embed_concurrency)
        self._upsert_buffer: list[qdrant_models.PointStruct] = []

    async def run(self) -> None:
        try:
            await self._ensure_collection()
            graph = AttackGraph()

            # Download/load each domain
            for domain in self.domains:
                bundle = await self._load_domain(domain)
                if bundle:
                    graph.ingest_bundle(bundle, domain)
                    logger.info("Loaded %s domain — %d total STIX objects", domain, len(graph.objects))

            stats = IngestStats()
            await self._build_and_upsert(graph, stats)
            await self._flush(stats)
            stats.log_summary()
            logger.info("MITRE ATT&CK ingestion complete.")
        finally:
            await self._qdrant.close()

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    async def _ensure_collection(self) -> None:
        if self.dry_run:
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

    # ------------------------------------------------------------------
    # Load domain JSON
    # ------------------------------------------------------------------

    async def _load_domain(self, domain: str) -> dict | None:
        if self.local_dir:
            path = self.local_dir / f"{domain}-attack.json"
            if path.exists():
                logger.info("Loading %s from local file %s", domain, path)
                return json.loads(path.read_text())
            logger.warning("Local file not found: %s — skipping.", path)
            return None

        url = DOMAIN_URLS.get(domain)
        if not url:
            logger.warning("Unknown domain '%s' — skipping.", domain)
            return None

        logger.info("Downloading %s ATT&CK from GitHub...", domain)
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as http:
            for attempt in range(3):
                try:
                    resp = await http.get(url)
                    resp.raise_for_status()
                    return resp.json()
                except Exception as exc:
                    logger.warning("Download attempt %d failed for %s: %s", attempt + 1, domain, exc)
                    await asyncio.sleep(5 * (attempt + 1))
        logger.error("Failed to download %s after 3 attempts.", domain)
        return None

    # ------------------------------------------------------------------
    # Build documents and queue upserts
    # ------------------------------------------------------------------

    async def _build_and_upsert(self, graph: AttackGraph, stats: IngestStats) -> None:
        # Techniques
        for obj in graph.active_objects_of_type(TECHNIQUE_TYPE):
            try:
                text = build_technique_doc(obj, graph)
                if not text.strip():
                    continue
                attack_id = _mitre_id(obj)
                point_id = str(uuid.UUID(bytes=hashlib.sha256(f"mitre:tech:{obj['id']}".encode()).digest()[:16]))
                payload = {
                    "text": text,
                    "object_type": "technique",
                    "attack_id": attack_id,
                    "name": _safe(obj.get("name")),
                    "tactics": _tactics(obj),
                    "platforms": _platforms(obj),
                    "is_subtechnique": obj.get("x-mitre-is-subtechnique", False),
                    "domain": obj.get("_domain", "enterprise"),
                    "source": "MITRE ATT&CK",
                    "ingest_date": TODAY,
                    "url": _mitre_url(obj),
                }
                self._upsert_buffer.append(
                    qdrant_models.PointStruct(id=point_id, vector=[0.0] * EMBEDDING_DIM, payload=payload)
                )
                stats.techniques += 1
                await self._maybe_flush(stats)
            except Exception as exc:
                stats.errors += 1
                logger.warning("Error building technique %s: %s", obj.get("id"), exc)

        # Groups
        for obj in graph.active_objects_of_type(GROUP_TYPE):
            try:
                text = build_group_doc(obj, graph)
                if not text.strip():
                    continue
                attack_id = _mitre_id(obj)
                aliases = [str(a) for a in obj.get("aliases", []) if str(a) != _safe(obj.get("name"))]
                point_id = str(uuid.UUID(bytes=hashlib.sha256(f"mitre:group:{obj['id']}".encode()).digest()[:16]))
                payload = {
                    "text": text,
                    "object_type": "group",
                    "attack_id": attack_id,
                    "name": _safe(obj.get("name")),
                    "aliases": aliases,
                    "domain": obj.get("_domain", "enterprise"),
                    "source": "MITRE ATT&CK",
                    "ingest_date": TODAY,
                    "url": _mitre_url(obj),
                }
                self._upsert_buffer.append(
                    qdrant_models.PointStruct(id=point_id, vector=[0.0] * EMBEDDING_DIM, payload=payload)
                )
                stats.groups += 1
                await self._maybe_flush(stats)
            except Exception as exc:
                stats.errors += 1
                logger.warning("Error building group %s: %s", obj.get("id"), exc)

        # Software (malware + tool)
        for obj in graph.active_objects_of_type(*SOFTWARE_TYPES):
            try:
                text = build_software_doc(obj, graph)
                if not text.strip():
                    continue
                attack_id = _mitre_id(obj)
                soft_type = obj.get("type", "malware")
                point_id = str(uuid.UUID(bytes=hashlib.sha256(f"mitre:soft:{obj['id']}".encode()).digest()[:16]))
                payload = {
                    "text": text,
                    "object_type": soft_type,
                    "attack_id": attack_id,
                    "name": _safe(obj.get("name")),
                    "platforms": _platforms(obj),
                    "domain": obj.get("_domain", "enterprise"),
                    "source": "MITRE ATT&CK",
                    "ingest_date": TODAY,
                    "url": _mitre_url(obj),
                }
                self._upsert_buffer.append(
                    qdrant_models.PointStruct(id=point_id, vector=[0.0] * EMBEDDING_DIM, payload=payload)
                )
                stats.software += 1
                await self._maybe_flush(stats)
            except Exception as exc:
                stats.errors += 1
                logger.warning("Error building software %s: %s", obj.get("id"), exc)

        # Mitigations
        for obj in graph.active_objects_of_type(MITIGATION_TYPE):
            try:
                attack_id = _mitre_id(obj)
                # Skip procedure-level course-of-action objects (no external MITRE ID)
                if not attack_id.startswith("M"):
                    continue
                text = build_mitigation_doc(obj)
                if not text.strip():
                    continue
                point_id = str(uuid.UUID(bytes=hashlib.sha256(f"mitre:mit:{obj['id']}".encode()).digest()[:16]))
                payload = {
                    "text": text,
                    "object_type": "mitigation",
                    "attack_id": attack_id,
                    "name": _safe(obj.get("name")),
                    "domain": obj.get("_domain", "enterprise"),
                    "source": "MITRE ATT&CK",
                    "ingest_date": TODAY,
                    "url": _mitre_url(obj),
                }
                self._upsert_buffer.append(
                    qdrant_models.PointStruct(id=point_id, vector=[0.0] * EMBEDDING_DIM, payload=payload)
                )
                stats.mitigations += 1
                await self._maybe_flush(stats)
            except Exception as exc:
                stats.errors += 1
                logger.warning("Error building mitigation %s: %s", obj.get("id"), exc)

    # ------------------------------------------------------------------
    # Flush + embed
    # ------------------------------------------------------------------

    async def _maybe_flush(self, stats: IngestStats) -> None:
        if len(self._upsert_buffer) >= self.upsert_batch_size:
            await self._flush(stats)

    async def _flush(self, stats: IngestStats | None = None) -> None:
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
        if stats:
            stats.points_upserted += len(points)

    async def _embed_all(self, texts: list[str]) -> list[list[float]]:
        batches = [texts[i : i + self.embed_batch_size] for i in range(0, len(texts), self.embed_batch_size)]
        results: list[list[float]] = []
        for group_start in range(0, len(batches), self.embed_concurrency):
            group = batches[group_start : group_start + self.embed_concurrency]
            group_results = await asyncio.gather(*[self._embed_batch(b) for b in group])
            for vecs in group_results:
                results.extend(vecs)
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ingest MITRE ATT&CK STIX data into OSIA Qdrant knowledge base",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--domains",
        nargs="+",
        default=["enterprise"],
        choices=["enterprise", "mobile", "ics"],
        help="ATT&CK domains to ingest.",
    )
    p.add_argument("--local-dir", default="", dest="local_dir", help="Local directory with <domain>-attack.json files.")
    p.add_argument("--dry-run", action="store_true", help="Build documents but skip Qdrant writes.")
    p.add_argument("--embed-batch-size", type=int, default=48, dest="embed_batch_size")
    p.add_argument("--embed-concurrency", type=int, default=3, dest="embed_concurrency")
    p.add_argument("--upsert-batch-size", type=int, default=64, dest="upsert_batch_size")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not HF_TOKEN:
        parser.error("HF_TOKEN not set in environment — required for embeddings.")

    logger.info(
        "Starting MITRE ATT&CK ingest | domains=%s dry_run=%s",
        args.domains,
        args.dry_run,
    )

    if args.dry_run:
        logger.warning("DRY RUN — no data will be written to Qdrant.")

    ingestor = MitreAttackIngestor(args)
    asyncio.run(ingestor.run())


if __name__ == "__main__":
    main()

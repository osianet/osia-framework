"""
OSIA Aesthetic Pack Generator
Generates logos, backgrounds, overlays, and desk badges using Venice AI.
All assets are written to assets/aesthetic/ with a manifest.json index.

Usage:
    uv run python scripts/generate_aesthetic_pack.py                  # full pack
    uv run python scripts/generate_aesthetic_pack.py --category logos
    uv run python scripts/generate_aesthetic_pack.py --category backgrounds
    uv run python scripts/generate_aesthetic_pack.py --category overlays
    uv run python scripts/generate_aesthetic_pack.py --category desk_badges
    uv run python scripts/generate_aesthetic_pack.py --resume         # skip existing
    uv run python scripts/generate_aesthetic_pack.py --dry-run        # print prompts only
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.intelligence.venice_image_client import VeniceImageClient  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("osia.aesthetic")

ROOT = Path(__file__).parent.parent
AESTHETIC_CFG = ROOT / "config" / "aesthetic.yaml"
OUT_DIR = ROOT / "assets" / "aesthetic"
MANIFEST_PATH = OUT_DIR / "manifest.json"

CATEGORIES = ["logos", "backgrounds", "overlays", "desk_badges"]


def load_config() -> dict:
    with open(AESTHETIC_CFG) as f:
        return yaml.safe_load(f)


# ── Prompt builders ────────────────────────────────────────────────────────────


def _base_style(cfg: dict) -> str:
    p = cfg["palette"]
    return (
        "Dark cinematic intelligence agency aesthetic. "
        f"Colour palette: deep void black {p['primary']['void_black']}, "
        f"forest signal green {p['primary']['signal_green']}, "
        f"amber {p['primary']['amber_alert']}. "
        "Grain texture, low-key directional lighting. "
        "No text, no words, no letters, no watermarks, no UI elements."
    )


def logo_prompts(cfg: dict) -> list[dict]:
    base = _base_style(cfg)
    tagline = cfg["identity"]["tagline"]
    motifs = ", ".join(cfg["visual_language"]["motifs"][:4])
    prompt = (
        f"Minimalist emblem / seal for a radical open-source intelligence agency. "
        f"Motifs: {motifs}. "
        f"Ethos: '{tagline}'. "
        f"Decolonial, anti-imperialist aesthetic — NOT Western eagle or corporate logo. "
        f"Circular seal format, intricate but readable at small sizes. "
        f"Dark background version. {base}"
    )
    assets = []
    for variant in cfg["asset_specs"]["logo"]["variants"]:
        for size in cfg["asset_specs"]["logo"]["sizes"]:
            assets.append(
                {
                    "id": f"logo_{variant['id']}_{size}",
                    "category": "logos",
                    "variant": variant["id"],
                    "prompt": prompt,
                    "width": size,
                    "height": size,
                    "filename": f"logo_{variant['id']}_{size}.png",
                    "tags": ["logo", "brand", variant["id"]],
                    "bg": variant["bg"],
                }
            )
    return assets


def background_prompts(cfg: dict) -> list[dict]:
    base = _base_style(cfg)
    specs = cfg["asset_specs"]["backgrounds"]
    assets = []
    for cat in specs["categories"]:
        for size_name, (w, h) in specs["sizes"].items():
            prompt = _bg_prompt(cat, cfg, base, w, h)
            assets.append(
                {
                    "id": f"bg_{cat['id']}_{size_name}",
                    "category": "backgrounds",
                    "subcategory": cat["id"],
                    "prompt": prompt,
                    "width": w,
                    "height": h,
                    "filename": f"bg_{cat['id']}_{size_name}.png",
                    "tags": ["background", cat["id"], size_name],
                    "bg": "dark",
                }
            )
    return assets


def _bg_prompt(cat: dict, cfg: dict, base: str, w: int, h: int) -> str:
    orientation = "landscape" if w > h else ("portrait" if h > w else "square")
    desc = cat["desc"]
    motifs = cfg["visual_language"]["motifs"]

    extras = {
        "hero": f"Primary brand hero image. {motifs[0]}, {motifs[1]}. Vast, atmospheric, {orientation}.",
        "data_overlay": f"Abstract network topology, node-and-edge graph, circuit traces. Subtle, dark, {orientation}.",
        "terrain": f"Topographic contour map, satellite land imagery, contested borders. {orientation}.",
        "archive": f"Aged classified document texture, redaction bars, typewriter font fragments, yellowed paper. {orientation}.",
        "ecological": f"Mycelium network, forest canopy, watershed aerial view. Non-human intelligence. {orientation}.",
    }
    return f"{desc}. {extras.get(cat['id'], '')} {base}"


def overlay_prompts(cfg: dict) -> list[dict]:
    assets = []
    for cat in cfg["asset_specs"]["overlays"]["categories"]:
        prompt = _overlay_prompt(cat)
        assets.append(
            {
                "id": f"overlay_{cat['id']}",
                "category": "overlays",
                "subcategory": cat["id"],
                "prompt": prompt,
                "width": 1920,
                "height": 1080,
                "filename": f"overlay_{cat['id']}.png",
                "tags": ["overlay", "transparent", cat["id"]],
                "bg": "transparent",
                "usage_note": cat["desc"],
            }
        )
    return assets


def _overlay_prompt(cat: dict) -> str:
    prompts = {
        "scan_lines": (
            "Transparent PNG overlay of subtle horizontal CRT scan lines. "
            "Very faint, dark, evenly spaced. Pure black background with slight transparency. "
            "No text, no objects, just scan line texture."
        ),
        "noise_grain": (
            "Transparent PNG overlay of analogue film grain / noise texture. "
            "Fine monochromatic grain, subtle, dark. No text, no objects."
        ),
        "vignette": (
            "Transparent PNG vignette overlay. Dark edges fading to transparent centre. "
            "Smooth radial gradient. No text, no objects."
        ),
        "grid": (
            "Transparent PNG overlay of a faint intelligence-map grid. "
            "Thin lines, dark green tint, very subtle. No text, no labels."
        ),
        "redaction_bars": (
            "Transparent PNG overlay of several horizontal black redaction bars "
            "at varying vertical positions, as seen on declassified documents. "
            "No text, just solid black bars on transparent background."
        ),
    }
    return prompts.get(cat["id"], cat["desc"])


def desk_badge_prompts(cfg: dict) -> list[dict]:
    base = _base_style(cfg)
    assets = []
    for slug in cfg["asset_specs"]["desk_badges"]["desks"]:
        desk_cfg = cfg["desk_aesthetics"].get(slug, {})
        motif = desk_cfg.get("motif", "intelligence analysis")
        mood = desk_cfg.get("mood", "analytical")
        accent = desk_cfg.get("accent", "amber_alert")
        accent_hex = cfg["palette"]["primary"].get(accent) or cfg["palette"]["accent"].get(accent, "#C8860A")
        name = slug.replace("-", " ").replace(" desk", "").replace(" directorate", "").title()
        prompt = (
            f"Circular emblem / badge for the '{name}' intelligence desk. "
            f"Motif: {motif}. Mood: {mood}. Accent colour: {accent_hex}. "
            f"Decolonial, anti-imperialist aesthetic. Intricate seal design, transparent background. "
            f"{base}"
        )
        assets.append(
            {
                "id": f"badge_{slug}",
                "category": "desk_badges",
                "desk": slug,
                "prompt": prompt,
                "width": 512,
                "height": 512,
                "filename": f"badge_{slug}.png",
                "tags": ["badge", "desk", slug],
                "bg": "transparent",
            }
        )
    return assets


# ── Generation ─────────────────────────────────────────────────────────────────


def build_asset_list(cfg: dict, category_filter: str | None) -> list[dict]:
    all_assets = logo_prompts(cfg) + background_prompts(cfg) + overlay_prompts(cfg) + desk_badge_prompts(cfg)
    if category_filter:
        all_assets = [a for a in all_assets if a["category"] == category_filter]
    return all_assets


async def generate_pack(
    category_filter: str | None = None,
    resume: bool = False,
    dry_run: bool = False,
) -> None:
    cfg = load_config()
    assets = build_asset_list(cfg, category_filter)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing manifest
    manifest: dict = {}
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)

    client = VeniceImageClient(model="flux-2-pro")
    generated = 0
    skipped = 0

    for asset in assets:
        out_path = OUT_DIR / asset["filename"]

        if resume and out_path.exists():
            logger.info("SKIP (exists): %s", asset["filename"])
            skipped += 1
            continue

        if dry_run:
            logger.info("DRY-RUN [%s] %s\n  → %s", asset["category"], asset["id"], asset["prompt"][:120])
            continue

        logger.info("Generating [%s] %s (%dx%d)...", asset["category"], asset["id"], asset["width"], asset["height"])
        try:
            await client.generate(
                prompt=asset["prompt"],
                width=asset["width"],
                height=asset["height"],
                output_path=out_path,
            )
            asset["path"] = str(out_path.relative_to(ROOT))
            asset["generated_at"] = datetime.now(UTC).isoformat()
            manifest[asset["id"]] = asset
            _save_manifest(manifest)
            generated += 1
            logger.info("✓ %s", asset["filename"])
        except Exception as e:
            logger.error("✗ %s: %s", asset["id"], e)

    if not dry_run:
        logger.info("Done — %d generated, %d skipped. Manifest: %s", generated, skipped, MANIFEST_PATH)


def _save_manifest(manifest: dict) -> None:
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


# ── CLI ────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate OSIA aesthetic asset pack")
    parser.add_argument("--category", choices=CATEGORIES, default=None)
    parser.add_argument("--resume", action="store_true", help="Skip already-generated files")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without generating")
    args = parser.parse_args()
    asyncio.run(generate_pack(args.category, args.resume, args.dry_run))


if __name__ == "__main__":
    main()

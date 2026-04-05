"""
Generate OSIA logo candidates via Venice AI.

Produces multiple prompt variants so you can pick the best result.
All candidates saved to assets/logos/ with a manifest.

Usage:
    uv run python scripts/generate_logo.py              # all variants
    uv run python scripts/generate_logo.py --variant 2  # single variant
    uv run python scripts/generate_logo.py --size 2048  # override size
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.intelligence.venice_image_client import VeniceImageClient  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("osia.logo")

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "assets" / "logos"

# Shared seal structure — keep what works from the original
SEAL_STRUCTURE = (
    "Circular official seal / emblem. "
    "Outer ring contains the text 'OPEN SOURCE INTELLIGENCE AGENCY' arced along the top "
    "and 'OSIA' centered at the bottom. 'EST. 2026' on a banner at the base of the central image. "
    "Two concentric border rings in the outer circle. "
    "Four small stars evenly spaced around the lower ring. "
)

# Shared aesthetic constraints
AESTHETIC = (
    "Colour palette: deep void black #0A0C0B background, forest signal green #1A3A2A, "
    "warm amber #C8860A metallic accents. "
    "NOT navy blue, NOT gold CIA palette, NOT American eagle, NOT Western imperial iconography. "
    "Decolonial, anti-imperialist aesthetic. "
    "Embossed metallic relief style, cinematic, photorealistic render. "
    "Clean white background behind the seal for easy extraction. "
    "No extra decorative elements outside the seal circle. "
)

# Central image variants — different takes on the core iconography
VARIANTS = [
    {
        "id": "v1_eye_network",
        "name": "All-Seeing Network Eye",
        "central_image": (
            "Central image: a stylised open eye with an iris made of interconnected network nodes "
            "and data-graph edges — the eye that watches power, not the people. "
            "Below the eye, two symmetrical branches of topographic contour lines curve outward "
            "like roots or watersheds, replacing the olive branch and arrows. "
            "Behind the eye, a hexagonal shield with subtle circuit-trace lines. "
        ),
    },
    {
        "id": "v2_compass_mycelium",
        "name": "Compass Rose & Mycelium",
        "central_image": (
            "Central image: an eight-point compass rose at the centre, its cardinal points "
            "extending into mycelium network tendrils that spread organically outward — "
            "representing non-human intelligence and ecological sovereignty. "
            "The compass is overlaid on a topographic contour map circle. "
            "Below, two symmetrical fern fronds curve outward as base elements. "
        ),
    },
    {
        "id": "v3_owl_circuit",
        "name": "Geometric Owl",
        "central_image": (
            "Central image: a highly geometric, low-poly stylised owl facing forward, "
            "wings slightly spread, body composed of circuit-board trace lines and node points. "
            "The owl sits before a hexagonal shield bearing a network node graph. "
            "Below the owl, two symmetrical topographic contour line branches curve outward. "
            "The owl represents wisdom, night vision, and counter-surveillance — "
            "watching power from the shadows. "
        ),
    },
    {
        "id": "v4_fist_signal",
        "name": "Raised Fist & Signal Wave",
        "central_image": (
            "Central image: a stylised raised fist at the centre, rendered in geometric line-art, "
            "surrounded by concentric radio signal / wifi wave arcs emanating outward — "
            "representing data sovereignty and intelligence for the people. "
            "The fist is overlaid on a circular network node graph. "
            "Below, two symmetrical root/branch elements curve outward as base elements. "
        ),
    },
    {
        "id": "v5_earth_grid",
        "name": "Earth Grid & Roots",
        "central_image": (
            "Central image: a stylised globe / earth viewed from above, rendered as a "
            "topographic grid with latitude/longitude lines, overlaid with network node connections "
            "linking points across the Global South. "
            "Below the globe, two symmetrical root systems extend downward and outward, "
            "grounding the globe in the land. "
            "Behind, a hexagonal shield with subtle circuit traces. "
        ),
    },
]


def build_prompt(variant: dict) -> str:
    return SEAL_STRUCTURE + variant["central_image"] + AESTHETIC


def _load_manifest() -> dict:
    manifest_path = OUT_DIR / "manifest.json"
    return json.loads(manifest_path.read_text()) if manifest_path.exists() else {}


def _save_manifest(manifest: dict) -> None:
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))


async def generate_logos(variant_filter: int | None, size: int, remove_bg: bool) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest()
    variants = VARIANTS if variant_filter is None else [VARIANTS[variant_filter - 1]]
    client = VeniceImageClient(model="flux-2-pro")

    for v in variants:
        filename = f"{v['id']}_{size}.png"
        out_path = OUT_DIR / filename
        prompt = build_prompt(v)

        logger.info("Generating variant '%s' (%dx%d)...", v["name"], size, size)
        try:
            img_bytes = await client.generate(prompt=prompt, width=size, height=size, output_path=out_path)
            entry: dict = {
                "id": v["id"],
                "name": v["name"],
                "filename": filename,
                "path": str(out_path.relative_to(ROOT)),
                "size": size,
                "prompt": prompt,
                "generated_at": datetime.now(UTC).isoformat(),
                "tags": ["logo", "candidate", v["id"]],
                "bg": "white",
                "usage": "logo candidate — review and promote chosen variant to assets/osia_logo_sm.png",
            }
            manifest[v["id"]] = entry
            _save_manifest(manifest)
            logger.info("✓ Saved: %s", out_path)

            if remove_bg:
                await _remove_bg_for(client, v["id"], img_bytes, size, manifest)
        except Exception as e:
            logger.error("✗ %s: %s", v["id"], e)

    logger.info("Done. Review candidates in %s", OUT_DIR)


async def remove_bg_existing(variant_filter: int | None, size: int) -> None:
    """Remove background from already-generated logo files."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest()
    variants = VARIANTS if variant_filter is None else [VARIANTS[variant_filter - 1]]
    client = VeniceImageClient()

    for v in variants:
        src_path = OUT_DIR / f"{v['id']}_{size}.png"
        if not src_path.exists():
            logger.warning("Source not found (generate first): %s", src_path)
            continue
        img_bytes = src_path.read_bytes()
        await _remove_bg_for(client, v["id"], img_bytes, size, manifest)


async def _remove_bg_for(
    client: VeniceImageClient, variant_id: str, img_bytes: bytes, size: int, manifest: dict
) -> None:
    transparent_filename = f"{variant_id}_{size}_transparent.png"
    transparent_path = OUT_DIR / transparent_filename
    logger.info("Removing background for %s...", variant_id)
    try:
        await client.remove_background(img_bytes, output_path=transparent_path)
        manifest[f"{variant_id}_transparent"] = {
            "id": f"{variant_id}_transparent",
            "source_id": variant_id,
            "filename": transparent_filename,
            "path": str(transparent_path.relative_to(ROOT)),
            "size": size,
            "generated_at": datetime.now(UTC).isoformat(),
            "tags": ["logo", "transparent", "candidate", variant_id],
            "bg": "transparent",
            "usage": "transparent logo — use for overlays, compositing, and dark backgrounds",
        }
        _save_manifest(manifest)
        logger.info("✓ Transparent: %s", transparent_path)
    except Exception as e:
        logger.error("✗ bg-remove %s: %s", variant_id, e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate OSIA logo candidates")
    parser.add_argument(
        "--variant",
        type=int,
        choices=range(1, len(VARIANTS) + 1),
        help=f"Single variant (1-{len(VARIANTS)}). Omit for all.",
    )
    parser.add_argument("--size", type=int, default=1024, help="Output size in pixels (default: 1024)")
    parser.add_argument("--remove-bg", action="store_true", help="Also remove background after generation")
    parser.add_argument(
        "--remove-bg-only", action="store_true", help="Remove background from existing files (skip generation)"
    )
    args = parser.parse_args()

    print("\nVariants:")
    for i, v in enumerate(VARIANTS, 1):
        print(f"  {i}. [{v['id']}] {v['name']}")
    print()

    if args.remove_bg_only:
        asyncio.run(remove_bg_existing(args.variant, args.size))
    else:
        asyncio.run(generate_logos(args.variant, args.size, args.remove_bg))


if __name__ == "__main__":
    main()

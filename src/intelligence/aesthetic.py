"""Shared helpers for loading OSIA aesthetic assets and per-desk theme config."""

import base64
import logging
from functools import lru_cache
from pathlib import Path

import yaml

logger = logging.getLogger("osia.aesthetic")

_REPO_ROOT = Path(__file__).parent.parent.parent
_AESTHETIC_CFG = _REPO_ROOT / "config" / "aesthetic.yaml"
_ASSETS_DIR = _REPO_ROOT / "assets"


@lru_cache(maxsize=1)
def _load_config() -> dict:
    if _AESTHETIC_CFG.exists():
        with open(_AESTHETIC_CFG) as f:
            return yaml.safe_load(f)
    return {}


def desk_accent_colour(desk_slug: str) -> str:
    """Return the hex accent colour for a desk, falling back to amber."""
    cfg = _load_config()
    desk_cfg = cfg.get("desk_aesthetics", {}).get(desk_slug, {})
    accent_key = desk_cfg.get("accent", "amber_alert")
    palette = cfg.get("palette", {})
    return palette.get("primary", {}).get(accent_key) or palette.get("accent", {}).get(accent_key) or "#C8860A"


def desk_motif(desk_slug: str) -> str:
    cfg = _load_config()
    return cfg.get("desk_aesthetics", {}).get(desk_slug, {}).get("motif", "")


def _load_image_b64(path: Path) -> str | None:
    if path.exists():
        return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()
    return None


def load_logo_b64() -> str | None:
    return _load_image_b64(_ASSETS_DIR / "osia_logo_sm.png")


def load_portrait_b64(desk_slug: str) -> str | None:
    return _load_image_b64(_ASSETS_DIR / "portraits" / f"{desk_slug}.png")


def load_desk_badge_b64(desk_slug: str) -> str | None:
    """Load the transparent desk badge from the aesthetic pack, if generated."""
    # Prefer transparent version, fall back to opaque
    for suffix in (f"badge_{desk_slug}_transparent.png", f"badge_{desk_slug}.png"):
        path = _ASSETS_DIR / "aesthetic" / suffix
        if path.exists():
            return _load_image_b64(path)
    return None


# Maps each desk to the background category that best fits its aesthetic motif.
_DESK_BG_CATEGORY: dict[str, str] = {
    "geopolitical-and-security-desk": "terrain",
    "cultural-and-theological-intelligence-desk": "archive",
    "science-technology-and-commercial-desk": "data_overlay",
    "human-intelligence-and-profiling-desk": "hero",
    "finance-and-economics-directorate": "archive",
    "cyber-intelligence-and-warfare-desk": "data_overlay",
    "information-warfare-desk": "hero",
    "environment-and-ecology-desk": "ecological",
    "the-watch-floor": "hero",
}


def desk_bg_category(desk_slug: str) -> str:
    """Return the background image category for a desk (hero/terrain/archive/data_overlay/ecological)."""
    return _DESK_BG_CATEGORY.get(desk_slug, "hero")


def load_desk_bg_b64(desk_slug: str, orientation: str = "landscape") -> str | None:
    """Load the background image for a desk as a data URI.

    Args:
        desk_slug: Desk identifier, or an explicit category name (hero, terrain, etc.).
        orientation: 'landscape' → desktop size, 'portrait' → portrait size.
    """
    size_key = "desktop" if orientation == "landscape" else "portrait"
    category = _DESK_BG_CATEGORY.get(desk_slug, desk_slug)  # allow passing category directly
    path = _ASSETS_DIR / "aesthetic" / f"bg_{category}_{size_key}.png"
    if path.exists():
        raw = path.read_bytes()
        return "data:image/png;base64," + base64.b64encode(raw).decode()
    return None

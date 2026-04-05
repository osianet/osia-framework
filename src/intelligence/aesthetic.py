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

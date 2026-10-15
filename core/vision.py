"""Figures and diagrams: extract from PDF/PPTX, describe with a vision model, keep them citable.

A figure becomes its own Unit (kind="figure") with the image saved under the course's figures dir.
  - vision model reachable -> the unit text is the model's description of what the figure shows
  - otherwise              -> the unit is still created and citable, with an explicit
                              "visual content not captured" flag (the tutor then says so instead of guessing)
Descriptions are cached next to the image (<hash>.txt) so re-ingesting never pays twice.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from . import llm

MIN_SIDE = 90  # px: skips bullets, logos and icons
NOT_CAPTURED = "visual content not captured (no vision model available when this file was ingested)"

DESCRIBE_PROMPT = (
    "You are describing a figure from course material for a student who cannot see it. In 2-4 sentences say: "
    "what kind of figure it is, what the axes/labels/series are, how the curves or elements behave (trends, "
    "minima, crossings, ordering), and the one-sentence takeaway. Report only what is visible; do not add outside facts."
)


def save_image(blob: bytes, ext: str, figure_dir: Path) -> Path | None:
    """Store an extracted image (content-addressed). Returns None for icons and degenerate images."""
    try:
        from PIL import Image
        import io

        im = Image.open(io.BytesIO(blob))
        w, h = im.size
        if min(w, h) < MIN_SIDE or max(w, h) / max(min(w, h), 1) > 12:
            return None
        if im.mode not in ("RGB", "RGBA", "L"):
            im = im.convert("RGB")
            buf = io.BytesIO()
            im.save(buf, "PNG")
            blob, ext = buf.getvalue(), "png"
    except Exception:
        return None
    figure_dir.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha1(blob).hexdigest()[:16] + "." + (ext or "png").lower().replace("jpeg", "jpg")
    path = figure_dir / name
    if not path.exists():
        path.write_bytes(blob)
    return path


def describe(path: Path, context: str = "") -> tuple[str, str]:
    """Return (description, mode) with mode 'vision' | 'cached' | 'none'."""
    cache = path.with_suffix(".txt")
    if cache.exists():
        return cache.read_text(encoding="utf-8"), "cached"
    if not llm.vision_available():
        return "", "none"
    try:
        user = [{"type": "text", "text": DESCRIBE_PROMPT + (f"\nText near the figure: {context[:300]}" if context else "")},
                llm.image_part(path)]
        text = llm.chat(
            [{"role": "user", "content": user}],
            model=__import__("os").environ.get("VISION_MODEL") or None,
            max_tokens=350,
            temperature=0.1,
        ).strip()
    except Exception:
        return "", "none"
    if text:
        cache.write_text(text, encoding="utf-8")
    return text, "vision"


def figure_text(location: str, description: str, nearby: str) -> str:
    near = f" Nearby text: {nearby.strip()[:200]}" if nearby.strip() else ""
    if description:
        return f"Figure ({location}). {description}{near}"
    return f"Figure ({location}) — {NOT_CAPTURED}.{near}"

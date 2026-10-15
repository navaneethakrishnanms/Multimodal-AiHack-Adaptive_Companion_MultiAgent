"""LLM client — OpenAI-compatible chat completions.

Configure via environment variables (see config.example.env):
  LLM_BASE_URL  e.g. https://api.openai.com/v1
  LLM_API_KEY   your key
  LLM_MODEL     e.g. gpt-4o-mini / deepseek-v4.1-flash
  LLM_EXTRA_HEADERS  optional JSON dict of extra headers (e.g. routing headers)
Optional second models (all fall back to LLM_MODEL when unset):
  VERIFY_MODEL  independent model that re-solves generated quiz questions (cross-model check)
  VISION_MODEL  image-capable model used to describe figures and diagrams
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
import urllib.request
import urllib.error


class LLMError(RuntimeError):
    pass


ROOT = Path(__file__).resolve().parent.parent


def load_env(path: Path | None = None) -> None:
    """Load KEY=VALUE pairs from .env into os.environ (existing variables win). Never prints values."""
    envf = path or ROOT / ".env"
    if os.environ.get("STUDY_NO_ENV") or not envf.exists():
        return
    for line in envf.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_dead_until = 0.0
_dead_reason = ""
HARD_FAIL_COOLDOWN = 120.0


def _trip(reason: str) -> None:
    """Circuit breaker: after a credential/quota failure stop calling for a while so every
    request degrades to the offline fallback instantly instead of retrying a dead endpoint."""
    global _dead_until, _dead_reason
    _dead_until, _dead_reason = time.time() + HARD_FAIL_COOLDOWN, reason


def reset() -> None:
    global _dead_until, _dead_reason
    _dead_until, _dead_reason = 0.0, ""


def configured() -> bool:
    return bool(os.environ.get("LLM_BASE_URL") and os.environ.get("LLM_MODEL"))


def status() -> dict:
    """Never includes the key; the UI uses this to show online/offline mode."""
    healthy = configured() and time.time() >= _dead_until
    return {
        "configured": configured(),
        "healthy": healthy,
        "reason": "" if healthy else (_dead_reason or "not configured"),
        "model": os.environ.get("LLM_MODEL", ""),
        "verify_model": os.environ.get("VERIFY_MODEL", ""),
        "vision_model": os.environ.get("VISION_MODEL", ""),
    }


def available() -> bool:
    """True when a text model is configured and has not just failed hard (bad key / no credits)."""
    return configured() and time.time() >= _dead_until


def vision_available() -> bool:
    return available() and bool(os.environ.get("VISION_MODEL") or os.environ.get("LLM_MODEL"))


def image_part(path: Path) -> dict:
    """OpenAI-style image content part from a local file (data URL)."""
    import base64
    import mimetypes

    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    b64 = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def _cfg(model_override: str | None = None) -> tuple[str, str, str, dict]:
    base = os.environ.get("LLM_BASE_URL", "").rstrip("/")
    key = os.environ.get("LLM_API_KEY", "")
    model = model_override or os.environ.get("LLM_MODEL", "")
    extra = {}
    raw = os.environ.get("LLM_EXTRA_HEADERS", "")
    if raw:
        try:
            extra = json.loads(raw)
        except json.JSONDecodeError:
            extra = {}
    if not base or not model:
        raise LLMError(
            "LLM_BASE_URL / LLM_MODEL not set. Copy config.example.env to .env and fill it."
        )
    return base, key, model, extra


def chat(
    messages: list[dict],
    *,
    temperature: float = 0.2,
    max_tokens: int = 1500,
    retries: int = 3,
    json_mode: bool = False,
    model: str | None = None,
) -> str:
    """Single chat completion. Returns the assistant text. `model` overrides LLM_MODEL for this call."""
    base, key, model, extra = _cfg(model)
    payload: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    headers = {"Content-Type": "application/json", "User-Agent": "study-companion/1.0"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    headers.update(extra)

    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                f"{base}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            choice = data["choices"][0]
            text = choice["message"].get("content") or ""
            if not text and choice["message"].get("reasoning_content"):
                # Some reasoning models spend the budget on reasoning; retry with more room.
                payload["max_tokens"] = min(payload["max_tokens"] * 2, 8000)
                raise LLMError("empty content (reasoning consumed budget)")
            if choice.get("finish_reason") == "length" and (json_mode or len(text) < 20):
                # Truncated output is useless for JSON callers — retry with more room.
                payload["max_tokens"] = min(payload["max_tokens"] * 2, 8000)
                raise LLMError("truncated output (finish_reason=length)")
            return text
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            # Some providers reject response_format unless the prompt says "json".
            # Drop the strict mode and retry — the tolerant parser still handles it.
            if "response_format" in payload and "json" in body.lower():
                payload.pop("response_format", None)
                last_err = e
                continue
            if e.code in (401, 403):  # bad/missing key: retrying cannot help
                _trip(f"credentials rejected (HTTP {e.code})")
                raise LLMError(f"LLM rejected the credentials (HTTP {e.code})") from None
            if e.code == 429 and any(k in body for k in ("insufficient_quota", "credit_balance", "billing")):
                _trip("no API credits left (HTTP 429)")
                raise LLMError("LLM account has no credits (HTTP 429)") from None
            last_err = e
            time.sleep(1.5 * (attempt + 1))
        except Exception as e:  # noqa: BLE001 — surface a clean error after retries
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise LLMError(f"LLM call failed after {retries} attempts: {last_err}")


def chat_json(
    messages: list[dict],
    *,
    temperature: float = 0.1,
    max_tokens: int = 2500,
    retries: int = 3,
    model: str | None = None,
) -> dict | list:
    """Chat completion parsed as JSON (tolerant: strips code fences)."""
    text = chat(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        retries=retries,
        json_mode=True,
        model=model,
    )
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[: -3]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # tolerant fallback: find first { ... last }
        start = min(
            (text.find(c) for c in "{[" if text.find(c) != -1), default=-1
        )
        end = max(text.rfind("}"), text.rfind("]"))
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise

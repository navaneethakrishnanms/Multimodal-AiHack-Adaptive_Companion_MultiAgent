"""LLM client — OpenAI-compatible chat completions.

Configure via environment variables (see config.example.env):
  LLM_BASE_URL  e.g. https://api.openai.com/v1
  LLM_API_KEY   your key
  LLM_MODEL     e.g. gpt-4o-mini / deepseek-v4.1-flash
  LLM_EXTRA_HEADERS  optional JSON dict of extra headers (e.g. routing headers)
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
import urllib.error


class LLMError(RuntimeError):
    pass


def _cfg() -> tuple[str, str, str, dict]:
    base = os.environ.get("LLM_BASE_URL", "").rstrip("/")
    key = os.environ.get("LLM_API_KEY", "")
    model = os.environ.get("LLM_MODEL", "")
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
) -> str:
    """Single chat completion. Returns the assistant text."""
    base, key, model, extra = _cfg()
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
) -> dict | list:
    """Chat completion parsed as JSON (tolerant: strips code fences)."""
    text = chat(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        retries=retries,
        json_mode=True,
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

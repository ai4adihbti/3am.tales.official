"""Shared helpers: logging, config, the Gemini client, and run context."""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent


def log(agent: str, msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {agent:<16} {msg}", flush=True)


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ffprobe_duration(path: str | Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


@dataclass
class Context:
    """Everything an agent might need, passed down the pipeline."""
    cfg: dict
    workdir: Path
    mock: bool = False
    dry_run: bool = False
    llm: "LLM" = None  # type: ignore[assignment]
    notes: dict = field(default_factory=dict)


class LLMError(RuntimeError):
    pass


class LLM:
    """Minimal Gemini REST client that always returns parsed JSON.

    Uses only the free tier. Falls back across models on 404/429/5xx.
    """

    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, cfg: dict, mock: bool = False):
        self.mock = mock
        self.key = os.getenv("GEMINI_API_KEY", "")
        first = os.getenv("GEMINI_MODEL") or cfg["llm"]["model"]
        self.models = [first] + [m for m in cfg["llm"].get("fallback_models", []) if m != first]

    def json(self, prompt: str, *, system: str = "", temperature: float = 0.9,
             mock: dict | list | None = None) -> dict | list:
        if self.mock:
            if mock is None:
                raise LLMError("No mock response supplied")
            return mock
        if not self.key:
            raise LLMError("GEMINI_API_KEY is not set")

        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "responseMimeType": "application/json",
            },
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}

        last_err = ""
        for model in self.models:
            for attempt in range(4):
                try:
                    r = requests.post(self.URL.format(model=model), params={"key": self.key},
                                      json=body, timeout=90)
                except requests.RequestException as e:
                    last_err = f"{model}: {e}"
                    time.sleep(5 * (attempt + 1))
                    continue
                if r.status_code == 200:
                    try:
                        return _parse_json(_extract_text(r.json()))
                    except (ValueError, KeyError) as e:
                        last_err = f"{model}: bad JSON ({e})"
                        continue  # ask again
                last_err = f"{model}: HTTP {r.status_code} {r.text[:300]}"
                if r.status_code in (400, 403):
                    raise LLMError(last_err)        # bad key or bad request — don't retry
                if r.status_code == 404:
                    break                           # model not available → next model
                # 429 / 5xx → back off, then retry (free tier has per-minute limits)
                time.sleep(min(60, 15 * (attempt + 1)))
        raise LLMError(f"All Gemini models failed. Last error: {last_err}")


def _extract_text(data: dict) -> str:
    cand = data["candidates"][0]
    parts = cand.get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    if not text:
        raise ValueError(f"empty response (finishReason={cand.get('finishReason')})")
    return text


def _parse_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"(\{.*\}|\[.*\])", text, re.S)
        if not m:
            raise
        return json.loads(m.group(1))


SAFETY_RULES = """Content rules (Instagram-safe, Gen Z audience that may include 16-17 year olds):
- Suspense, dread and atmosphere over gore. No graphic injuries, torture or body horror detail.
- No suicide, self-harm, eating disorders, sexual content, or harm to children/animals.
- No real, named private individuals. No real tragedies or victims used for entertainment.
- No hate, slurs or stereotypes. No instructions for anything dangerous.
- Fiction formats are clearly fiction; the 'real mystery' format must be accurate and hedged."""

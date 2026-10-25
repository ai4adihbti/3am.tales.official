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

        errors: list[str] = []
        for model in self.models:
            # Cap "thinking" so replies come back quickly; some models don't accept
            # the setting, in which case it's dropped and the request is retried.
            use_thinking_cap = True
            attempt = 0
            while attempt < 3:
                body = {
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": temperature,
                                         "responseMimeType": "application/json"},
                }
                if use_thinking_cap:
                    body["generationConfig"]["thinkingConfig"] = {"thinkingBudget": 512}
                if system:
                    body["systemInstruction"] = {"parts": [{"text": system}]}

                attempt += 1
                started = time.time()
                try:
                    r = requests.post(self.URL.format(model=model), params={"key": self.key},
                                      json=body, timeout=(15, 180))
                except requests.Timeout:
                    errors.append(f"{model} #{attempt}: no reply after {time.time() - started:.0f}s")
                    log("Gemini", errors[-1])
                    continue
                except requests.RequestException as e:
                    errors.append(f"{model} #{attempt}: network error {type(e).__name__}")
                    log("Gemini", errors[-1])
                    time.sleep(5 * attempt)
                    continue

                if r.status_code == 200:
                    try:
                        return _parse_json(_extract_text(r.json()))
                    except (ValueError, KeyError) as e:
                        errors.append(f"{model} #{attempt}: unusable reply ({e})")
                        log("Gemini", errors[-1])
                        continue

                msg = r.text[:300].replace("\n", " ")
                errors.append(f"{model} #{attempt}: HTTP {r.status_code} {msg}")
                log("Gemini", errors[-1])
                if r.status_code == 400 and use_thinking_cap and "thinking" in msg.lower():
                    use_thinking_cap = False
                    attempt -= 1                     # not a real failure; retry without the cap
                    continue
                if r.status_code in (401, 403) or (r.status_code == 400 and "api key" in msg.lower()):
                    raise LLMError("Gemini rejected the API key. Check the GEMINI_API_KEY secret. "
                                   f"Details: {msg}")
                if r.status_code in (400, 404):
                    break                            # model unavailable or unsupported → next model
                # 429 rate limit / 5xx overload → wait, then retry
                time.sleep(min(60, 20 * attempt))
        raise LLMError("All Gemini models failed:\n  " + "\n  ".join(errors))


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
- No hate, slurs or stereotypes. No mocking of any religion, caste, region or community;
  folklore is treated as folklore, never as a slur against real people or practices.
- No instructions for anything dangerous (no rituals people could copy to "summon" things).
- All stories are fiction."""

HINGLISH_GUIDE = """Language: natural spoken Hinglish the way Indian Gen Z actually talks — about 75% Hindi,
with everyday English words mixed in (phone, call, battery, hostel, PG, seriously, sorry, bro, scene, network,
room, lift, chill, weird). Casual, not textbook or shuddh Hindi, not filmy. Light slang is fine (yaar, bhai,
abey, kya bakwaas, sach mein?, pagal hai kya). Short lines, interruptions, half-sentences, real reactions."""

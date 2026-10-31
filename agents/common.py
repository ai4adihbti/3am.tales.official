"""Shared helpers: logging, config, the Gemini client, and run context."""
from __future__ import annotations

import base64
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

    Uses only the free tier. When a model is rate-limited it immediately moves on to the next
    one and keeps using whichever model is working for the rest of the run. Models that are
    retired or have no quota left today are dropped. Only when every model is busy does it wait.
    """

    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, cfg: dict, mock: bool = False):
        self.mock = mock
        self.key = os.getenv("GEMINI_API_KEY", "")
        first = os.getenv("GEMINI_MODEL") or cfg["llm"]["model"]
        self.models = [first] + [m for m in cfg["llm"].get("fallback_models", []) if m != first]
        # Groq (free, USA) as a backup writer: used when Gemini's strong models are busy, before Gemini "lite"
        self.groq_key = os.getenv("GROQ_API_KEY", "").strip()
        if self.groq_key:
            groq = ["groq:" + m for m in cfg["llm"].get("groq_models", ["openai/gpt-oss-120b", "openai/gpt-oss-20b"])]
            lite_at = next((i for i, m in enumerate(self.models) if "lite" in m), len(self.models))
            self.models[lite_at:lite_at] = groq
        self.no_thinking_cap: set[str] = set()
        self.calls = 0          # successful requests (for the monthly spend estimate)
        self.dead_reasons: list[str] = []
        self._cfg_llm = cfg.get("llm", {})
        self.slow: set[str] = set()

    def _groq(self, model: str, prompt: str, system: str, temperature: float, errors: list[str],
              max_tokens: int = 16000):
        """Groq's OpenAI-compatible API, JSON mode. Returns (result, kind) like _call."""
        name = model.split(":", 1)[1]
        msgs = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt + "\n\nReply with valid JSON only."}]
        body = {"model": name, "messages": msgs, "temperature": min(temperature, 1.0),
                "response_format": {"type": "json_object"}, "max_completion_tokens": max_tokens}
        if max_tokens < 16000 and "gpt-oss" in name:
            body["reasoning_effort"] = "low"          # little answer room: spend it on the answer, not on thinking
        try:
            r = requests.post("https://api.groq.com/openai/v1/chat/completions", timeout=(15, 180),
                              headers={"Authorization": f"Bearer {self.groq_key}"}, json=body)
        except requests.RequestException as e:
            errors.append(f"{model}: network error {type(e).__name__}")
            return None, "busy"
        if r.status_code == 413 and max_tokens == 16000:
            # free plan: prompt + room for the answer must fit the per-minute token limit; ask for a shorter answer room
            m = re.search(r"Limit\D*(\d+)\D+Requested\D*(\d+)", r.text)
            if m:
                limit, requested = int(m.group(1)), int(m.group(2))
                room = limit - (requested - max_tokens) - 300
                # 1,500 tokens still fit a review or a caption (a too-short screenplay just fails over to the next model)
                if room >= 1500:
                    return self._groq(model, prompt, system, temperature, errors, max_tokens=room)
        if r.status_code == 200:
            try:
                return _parse_json(r.json()["choices"][0]["message"]["content"]), "ok"
            except (ValueError, KeyError, IndexError, TypeError) as e:
                errors.append(f"{model}: unusable reply ({e})")
                return None, "busy"
        errors.append(f"{model}: HTTP {r.status_code} {r.text[:200]}")
        if r.status_code in (401, 403):
            log("Gemini", f"{model}: Groq rejected the key — check the GROQ_API_KEY secret; skipping Groq")
            return None, "gone"
        if r.status_code == 404:                     # model retired by Groq: don't ask it again this run
            log("Gemini", f"{model}: no longer offered by Groq (HTTP 404) — dropped for this run")
            return None, "gone"
        if r.status_code in (400, 413):              # this prompt is too big for its free limits
            log("Gemini", f"{model}: not usable for this request (HTTP {r.status_code})")
            return None, "busy"
        log("Gemini", f"{model}: busy (HTTP {r.status_code}) — trying the next model")
        return None, "busy"

    def _call(self, model: str, prompt: str, system: str, temperature: float, errors: list[str],
              images: list | None = None):
        """One model, at most 2 tries. Returns (result, kind): ok | busy | gone | key."""
        if model.startswith("groq:"):
            if images:
                return None, "busy"                    # Groq is text-only here; picture checks stay on Gemini
            return self._groq(model, prompt, system, temperature, errors)
        for attempt in (1, 2):
            parts = [{"inline_data": {"mime_type": m, "data": base64.b64encode(b).decode()}} for m, b in (images or [])]
            body = {"contents": [{"role": "user", "parts": parts + [{"text": prompt}]}],
                    "generationConfig": {"temperature": temperature, "responseMimeType": "application/json",
                                         "maxOutputTokens": 24000},
                    "safetySettings": [{"category": c, "threshold": "BLOCK_ONLY_HIGH"} for c in (
                        "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                        "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")]}
            if model not in self.no_thinking_cap:
                body["generationConfig"]["thinkingConfig"] = {"thinkingBudget": 512}
            if system:
                body["systemInstruction"] = {"parts": [{"text": system}]}
            started = time.time()
            try:
                r = requests.post(self.URL.format(model=model), params={"key": self.key},
                                  json=body, timeout=(15, 180))
            except requests.Timeout:
                errors.append(f"{model}: no reply after {time.time() - started:.0f}s")
                if model not in self.slow:
                    log("Gemini", errors[-1] + " — trying other models first for the rest of this run")
                else:
                    log("Gemini", errors[-1])
                self.slow.add(model)
                return None, "busy"
            except requests.RequestException as e:
                errors.append(f"{model}: network error {type(e).__name__}")
                log("Gemini", errors[-1])
                time.sleep(5)
                continue

            if r.status_code == 200:
                try:
                    data = r.json()
                except ValueError:
                    data = {}
                reason = _block_reason(data)
                if reason:
                    errors.append(f"{model}: blocked by Gemini's safety filter ({reason})")
                    log("Gemini", errors[-1])
                    return reason, "blocked"
                try:
                    return _parse_json(_extract_text(r.json())), "ok"
                except (ValueError, KeyError) as e:
                    errors.append(f"{model}: unusable reply ({e})")
                    log("Gemini", errors[-1])
                    continue  # ask once more

            msg = r.text[:240].replace("\n", " ")
            errors.append(f"{model}: HTTP {r.status_code} {msg}")
            if r.status_code == 400 and "thinking" in msg.lower() and model not in self.no_thinking_cap:
                self.no_thinking_cap.add(model)
                continue  # retry without the thinking cap
            if r.status_code in (401, 403) or (r.status_code == 400 and "api key" in msg.lower()):
                return msg, "key"
            if r.status_code in (400, 404) or is_hard_quota(r):
                log("Gemini", f"{model}: not usable today (HTTP {r.status_code}) — dropped for this run")
                if r.status_code == 429:
                    self.dead_reasons.append("quota")
                return None, "gone"
            if attempt == 1 and model == self.models[0] and model not in self.slow:
                log("Gemini", f"{model}: busy (HTTP {r.status_code}) — waiting 20s and trying again")
                time.sleep(20)
                continue
            log("Gemini", f"{model}: busy (HTTP {r.status_code}) — trying the next model")
            return None, "busy"  # 429 per-minute limit or 5xx overload
        return None, "busy"

    def json(self, prompt: str, *, system: str = "", temperature: float = 0.9,
             mock: dict | list | None = None, images: list | None = None, list_key: str = "") -> dict:
        """`images`: optional list of (mime_type, bytes) sent along with the prompt.
        Always returns a dict: if a model answers with a bare JSON list, it is wrapped as {list_key: [...]}
        (or, without list_key, its first object is used)."""
        out = self._json(prompt, system=system, temperature=temperature, mock=mock, images=images)
        if isinstance(out, list):
            if list_key:
                return {list_key: out}
            return next((x for x in out if isinstance(x, dict)), {})
        return out if isinstance(out, dict) else {}

    def _json(self, prompt: str, *, system: str = "", temperature: float = 0.9,
              mock: dict | list | None = None, images: list | None = None) -> dict | list:
        if self.mock:
            if mock is None:
                raise LLMError("No mock response supplied")
            return mock
        if not self.key:
            raise LLMError("GEMINI_API_KEY is not set")

        errors: list[str] = []
        patience = int(self._cfg_llm.get("patience_rounds", 2))
        for rnd in range(4 + patience):
            strong = [m for m in self.models if "lite" not in m]
            # when Google is overloaded, wait for a strong model a little before settling for a "lite" one:
            # lite models write noticeably weaker stories
            pool = strong if (rnd < patience and strong) else list(self.models)
            # models that already failed to answer in time this run are tried last (saves ~3 min per call)
            pool = [m for m in pool if m not in self.slow] + [m for m in pool if m in self.slow]
            for model in pool:
                result, kind = self._call(model, prompt, system, temperature, errors, images)
                if kind == "ok":
                    self.calls += 1
                    if model != self.models[0]:
                        log("Gemini", f"answered by backup model {model}")
                    return result
                if kind == "blocked":
                    raise LLMBlocked(f"Gemini's safety filter blocked this request ({result})")
                if kind == "key":
                    raise LLMError(f"Gemini rejected the API key. Check the GEMINI_API_KEY secret. Details: {result}")
                if kind == "gone":
                    self.models.remove(model)
            if not self.models:
                break
            wait = 45 if rnd < patience else 30 * (rnd - patience + 1)
            log("Gemini", f"{'strong models' if pool is strong and rnd < patience else 'all models'} busy — "
                          f"waiting {wait}s before trying again")
            time.sleep(wait)

        if not self.models and self.dead_reasons:
            raise LLMError("The free Gemini quota is used up for today on every model. It resets around "
                           "1:30 PM IST (midnight US Pacific). Nothing is wrong with your setup; "
                           "run it again after that.\n  " + "\n  ".join(errors[-6:]))
        raise LLMError("All Gemini models failed:\n  " + "\n  ".join(errors[-8:]))


def is_hard_quota(r: requests.Response) -> bool:
    """A 429 that won't clear by waiting a minute: a daily cap, or a model with no free quota."""
    if r.status_code != 429:
        return False
    t = r.text
    return "PerDay" in t or "per day" in t.lower() or re.search(r"limit:\s*0\b", t) is not None


class LLMBlocked(LLMError):
    """The prompt or reply was blocked by Gemini's safety filter (retrying the same prompt won't help)."""


def _block_reason(data: dict) -> str:
    fb = (data.get("promptFeedback") or {}).get("blockReason")
    if fb:
        return str(fb)
    cands = data.get("candidates")
    if cands is None and data:
        return "no reply (blocked)"
    if cands:
        fr = cands[0].get("finishReason", "")
        parts = (cands[0].get("content") or {}).get("parts") or []
        if fr in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII") and not any(p.get("text") for p in parts):
            return fr
    return ""


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
    candidates = [text]
    m = re.search(r"(\{.*\}|\[.*\])", text, re.S)
    if m:
        candidates.append(m.group(1))
    for c in list(candidates):
        fixed = re.sub(r",\s*([}\]])", r"\1", c)                          # trailing commas
        fixed = fixed.replace("\u201c", '"').replace("\u201d", '"')      # smart quotes around keys
        candidates.append(fixed)
    last = None
    for c in candidates:
        try:
            return json.loads(c)
        except json.JSONDecodeError as e:
            last = e
    raise last


SAFETY_RULES = """Content rules (Instagram-safe, Gen Z audience that may include 16-17 year olds):
- Suspense, dread and atmosphere over gore. No graphic injuries, torture or body horror detail.
- No suicide, self-harm, eating disorders, sexual content, or harm to children/animals.
- No real, named private individuals. No real tragedies or victims used for entertainment.
- No hate, slurs or stereotypes. No mocking of any religion, caste, region or community;
  folklore is treated as folklore, never as a slur against real people or practices.
- No instructions for anything dangerous (no rituals people could copy to "summon" things).
- All stories are fiction."""

HINGLISH_GUIDE = """Language: natural spoken Hindi the way real Indians talk — Hindi-heavy, with the everyday English
words people actually use (phone, network, battery, hospital, bus, sorry, please, seriously). Not textbook or
shuddh Hindi, not filmy dialogue-baazi. Match each character: young friends use light slang (yaar, bhai, abey,
pagal hai kya); a mother speaks warm, simple Hindi (beta, chal, darr mat); a child speaks in short innocent
sentences; an elder speaks slowly with old-fashioned words. Short lines, interruptions, half-sentences, real
reactions."""

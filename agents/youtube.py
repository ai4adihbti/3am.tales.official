"""YouTube agent: uploads the same reel to the channel as a YouTube Short.

Needs three GitHub secrets (set up once, see README): YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN.
A vertical video of 3:00 or less is a Short; the system keeps reels at or under 2:55 for that reason.
If anything fails here, the Instagram post is unaffected.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import requests

from .common import Context, ffprobe_duration, log

NAME = "YouTube"


class YouTubeError(RuntimeError):
    pass


def ready() -> bool:
    return all(os.getenv(k) for k in ("YT_CLIENT_ID", "YT_CLIENT_SECRET", "YT_REFRESH_TOKEN"))


def _access_token() -> str:
    r = requests.post("https://oauth2.googleapis.com/token", timeout=30, data={
        "client_id": os.getenv("YT_CLIENT_ID", "").strip(),
        "client_secret": os.getenv("YT_CLIENT_SECRET", "").strip(),
        "refresh_token": os.getenv("YT_REFRESH_TOKEN", "").strip(),
        "grant_type": "refresh_token"})
    if r.status_code != 200:
        hint = ""
        if "invalid_grant" in r.text:
            hint = (" — the YouTube permission was revoked or expired. Redo the OAuth Playground step and update "
                    "the YT_REFRESH_TOKEN secret (check the Google app is 'In production', not 'Testing').")
        elif "invalid_client" in r.text:
            hint = " — check the YT_CLIENT_ID and YT_CLIENT_SECRET secrets."
        raise YouTubeError(f"login failed: HTTP {r.status_code} {r.text[:160]}{hint}")
    return r.json()["access_token"]


def _metadata(ctx: Context, series: dict, caption: str) -> dict:
    yt = ctx.cfg.get("youtube", {})
    title_core = str(series.get("title") or "3AM Tale")
    total = int(series.get("total_parts") or 1)
    part = f" Part {series.get('part', 1)}" if total > 1 else ""
    title = f"{title_core}{part} 😨 | Hindi Horror Story #shorts"
    if len(title) > 100:
        title = f"{title_core[:60]}{part} | Hindi Horror #shorts"[:100]
    handle = ctx.cfg["channel"].get("watermark", "")
    body = caption.replace("Follow karo", "Subscribe karo").replace(handle, "")
    body = re.sub(r"\s*#aistory\b", "", body, flags=re.I).strip()
    desc = (f"{body}\n\n🕯️ Roz raat ek nayi bhootiya kahani — Subscribe karo!\n"
            f"Instagram: {handle}\nAll stories are fiction.\n#shorts")
    tags = re.findall(r"#(\w+)", caption) + list(yt.get("tags", []))
    tags = list(dict.fromkeys(t for t in tags if t.lower() != "aistory"))
    keep, size = [], 0
    for t in tags:                                   # YouTube allows ~500 characters of tags
        if size + len(t) + 2 > 450:
            break
        keep.append(t)
        size += len(t) + 2
    return {"snippet": {"title": title, "description": desc[:4900].replace("<", "").replace(">", ""),
                        "tags": keep, "categoryId": str(yt.get("category_id", "24")),
                        "defaultLanguage": "hi", "defaultAudioLanguage": "hi"},
            "status": {"privacyStatus": yt.get("privacy", "public"), "selfDeclaredMadeForKids": False,
                       "containsSyntheticMedia": bool(yt.get("contains_synthetic_media", False))}}


def upload(ctx: Context, video: Path, series: dict, caption: str) -> dict:
    if not ready():
        return {}
    secs = ffprobe_duration(video)
    if secs > 180:
        log(NAME, f"note: this reel is {secs:.0f}s (over 3:00), so YouTube will show it as a normal video, not a Short")
    token = _access_token()
    meta = _metadata(ctx, series, caption)
    size = video.stat().st_size
    r = requests.post("https://www.googleapis.com/upload/youtube/v3/videos", timeout=60,
                      params={"uploadType": "resumable", "part": "snippet,status"},
                      headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=UTF-8",
                               "X-Upload-Content-Type": "video/mp4", "X-Upload-Content-Length": str(size)},
                      data=json.dumps(meta))
    if r.status_code != 200 or "Location" not in r.headers:
        msg = r.text[:240]
        if "quotaExceeded" in msg or "uploadLimitExceeded" in msg:
            raise YouTubeError(f"YouTube's daily upload limit reached; try tomorrow ({msg[:120]})")
        if "youtubeSignupRequired" in msg:
            raise YouTubeError("this Google account has no YouTube channel yet — create the 3AM Tales channel first")
        raise YouTubeError(f"couldn't start the upload: HTTP {r.status_code} {msg}")
    with open(video, "rb") as fh:
        up = requests.put(r.headers["Location"], data=fh, timeout=1800,
                          headers={"Authorization": f"Bearer {token}", "Content-Type": "video/mp4"})
    if up.status_code not in (200, 201):
        raise YouTubeError(f"upload failed: HTTP {up.status_code} {up.text[:200]}")
    vid = up.json().get("id", "")
    url = f"https://youtube.com/shorts/{vid}"
    log(NAME, f"UPLOADED ✓ {url} ({meta['snippet']['title']})")
    return {"youtube_id": vid, "youtube_url": url}

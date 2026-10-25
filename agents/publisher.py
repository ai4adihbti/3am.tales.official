"""Publisher agent: posts the Reel through the official Instagram API.

Uses resumable upload, so the video goes straight from this machine to Instagram;
there's no need to host it at a public URL first.
Flow: create REELS container → upload bytes → wait for processing → publish.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import requests

from .common import Context, log

NAME = "Publisher"


class PublishError(RuntimeError):
    pass


def _check(r: requests.Response, step: str) -> dict:
    try:
        data = r.json()
    except ValueError:
        data = {"raw": r.text[:300]}
    if r.status_code >= 400 or "error" in data:
        raise PublishError(f"{step} failed: HTTP {r.status_code} {data}")
    return data


def publish_reel(ctx: Context, video: Path, caption: str) -> dict:
    ig = ctx.cfg["instagram"]
    token = os.getenv("IG_ACCESS_TOKEN", "")
    user_id = os.getenv("IG_USER_ID", "")
    if not token or not user_id:
        raise PublishError("IG_ACCESS_TOKEN and IG_USER_ID must be set")
    host = os.getenv("IG_API_HOST") or ig["api_host"]
    ver = ig["api_version"]
    graph = f"https://{host}/{ver}"

    # 1) container
    r = requests.post(f"{graph}/{user_id}/media", data={
        "media_type": "REELS", "upload_type": "resumable", "caption": caption,
        "share_to_feed": "true", "access_token": token}, timeout=60)
    container = _check(r, "create container")["id"]
    log(NAME, f"container {container} created")

    # 2) upload bytes
    size = video.stat().st_size
    with open(video, "rb") as fh:
        r = requests.post(f"https://rupload.facebook.com/ig-api-upload/{ver}/{container}",
                          headers={"Authorization": f"OAuth {token}", "offset": "0",
                                   "file_size": str(size)}, data=fh, timeout=600)
    _check(r, "upload video")
    log(NAME, f"uploaded {size / 1e6:.1f} MB")

    # 3) wait for Instagram to process the video (usually 30 s – 3 min)
    deadline = time.time() + 15 * 60
    status = ""
    while time.time() < deadline:
        r = requests.get(f"{graph}/{container}", params={
            "fields": "status_code,status", "access_token": token}, timeout=30)
        data = _check(r, "check status")
        status = data.get("status_code", "")
        if status == "FINISHED":
            break
        if status in ("ERROR", "EXPIRED"):
            raise PublishError(f"Instagram could not process the video: {data}")
        time.sleep(10)
    if status != "FINISHED":
        raise PublishError("Timed out waiting for Instagram to process the video")
    log(NAME, "processing finished")

    # 4) publish
    r = requests.post(f"{graph}/{user_id}/media_publish",
                      data={"creation_id": container, "access_token": token}, timeout=60)
    media_id = _check(r, "publish")["id"]

    permalink = ""
    try:
        r = requests.get(f"{graph}/{media_id}", params={"fields": "permalink", "access_token": token}, timeout=30)
        permalink = r.json().get("permalink", "")
    except requests.RequestException:
        pass
    log(NAME, f"PUBLISHED ✓ {permalink or media_id}")
    return {"media_id": media_id, "permalink": permalink}

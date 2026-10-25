"""Publisher agent: posts the Reel through the official Instagram API.

Route 1 (preferred): resumable upload, so the video goes straight from this machine to Instagram.
Route 2 (automatic backup): if Instagram doesn't allow route 1 for your login type, the video is
put on a temporary file host for 24 hours and Instagram fetches it from there (`video_url`).
Either way: create REELS container → wait for processing → publish.
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
    if r.status_code >= 400 or (isinstance(data, dict) and "error" in data):
        raise PublishError(f"{step} failed: HTTP {r.status_code} {data}")
    return data


# ── route 1: direct upload ──────────────────────────────────────────────
def _container_resumable(graph: str, ver: str, user_id: str, token: str, video: Path, caption: str) -> str:
    r = requests.post(f"{graph}/{user_id}/media", data={
        "media_type": "REELS", "upload_type": "resumable", "caption": caption,
        "share_to_feed": "true", "access_token": token}, timeout=60)
    container = _check(r, "create container (direct upload)")["id"]
    size = video.stat().st_size
    with open(video, "rb") as fh:
        r = requests.post(f"https://rupload.facebook.com/ig-api-upload/{ver}/{container}",
                          headers={"Authorization": f"OAuth {token}", "offset": "0",
                                   "file_size": str(size)}, data=fh, timeout=900)
    _check(r, "upload video")
    log(NAME, f"uploaded {size / 1e6:.1f} MB directly to Instagram")
    return container


# ── route 2: temporary public URL ───────────────────────────────────────
def _host_litterbox(video: Path) -> str:
    with open(video, "rb") as fh:
        r = requests.post("https://litterbox.catbox.moe/resources/internals/api.php",
                          data={"reqtype": "fileupload", "time": "24h"},
                          files={"fileToUpload": (video.name, fh, "video/mp4")}, timeout=900)
    url = r.text.strip()
    if r.status_code != 200 or not url.startswith("https://"):
        raise PublishError(f"litterbox: HTTP {r.status_code} {r.text[:120]}")
    return url


def _host_tmpfiles(video: Path) -> str:
    with open(video, "rb") as fh:
        r = requests.post("https://tmpfiles.org/api/v1/upload",
                          files={"file": (video.name, fh, "video/mp4")}, timeout=900)
    url = _check(r, "tmpfiles")["data"]["url"]
    return url.replace("tmpfiles.org/", "tmpfiles.org/dl/", 1).replace("http://", "https://", 1)


def _container_url(graph: str, user_id: str, token: str, video: Path, caption: str) -> str:
    errors = []
    for host in (_host_litterbox, _host_tmpfiles):
        try:
            url = host(video)
        except (requests.RequestException, PublishError, KeyError) as e:
            errors.append(str(e))
            continue
        log(NAME, f"video hosted temporarily via {host.__name__[6:]}")
        r = requests.post(f"{graph}/{user_id}/media", data={
            "media_type": "REELS", "video_url": url, "caption": caption,
            "share_to_feed": "true", "access_token": token}, timeout=120)
        try:
            return _check(r, "create container (video_url)")["id"]
        except PublishError as e:
            errors.append(str(e))
    raise PublishError("Backup upload route failed: " + " | ".join(errors))


def _wait(graph: str, container: str, token: str) -> None:
    deadline = time.time() + 20 * 60
    while time.time() < deadline:
        r = requests.get(f"{graph}/{container}", params={
            "fields": "status_code,status", "access_token": token}, timeout=30)
        data = _check(r, "check status")
        status = data.get("status_code", "")
        if status == "FINISHED":
            return
        if status in ("ERROR", "EXPIRED"):
            raise PublishError(f"Instagram could not process the video: {data}")
        time.sleep(10)
    raise PublishError("Timed out waiting for Instagram to process the video")


def publish_reel(ctx: Context, video: Path, caption: str) -> dict:
    ig = ctx.cfg["instagram"]
    token = os.getenv("IG_ACCESS_TOKEN", "")
    user_id = os.getenv("IG_USER_ID", "")
    if not token or not user_id:
        raise PublishError("IG_ACCESS_TOKEN and IG_USER_ID must be set")
    host = os.getenv("IG_API_HOST") or ig["api_host"]
    ver = ig["api_version"]
    graph = f"https://{host}/{ver}"

    try:
        container = _container_resumable(graph, ver, user_id, token, video, caption)
    except (PublishError, requests.RequestException) as e:
        log(NAME, f"direct upload not available ({str(e)[:160]}); using backup route")
        container = _container_url(graph, user_id, token, video, caption)
    log(NAME, f"container {container} — waiting for Instagram to process the video")

    _wait(graph, container, token)
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

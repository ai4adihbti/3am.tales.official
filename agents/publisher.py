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
# Instagram downloads the video from a public link. Several free hosts are tried in turn; each link is
# checked to really serve the MP4 before Instagram is given it, and if Instagram can't process the video
# from one host, the next host is tried.
def _host_catbox(video: Path) -> str:
    with open(video, "rb") as fh:
        r = requests.post("https://catbox.moe/user/api.php", data={"reqtype": "fileupload"},
                          files={"fileToUpload": (video.name, fh, "video/mp4")}, timeout=900)
    url = r.text.strip()
    if r.status_code != 200 or not url.startswith("https://"):
        raise PublishError(f"catbox: HTTP {r.status_code} {r.text[:120]}")
    return url


def _host_litterbox(video: Path) -> str:
    with open(video, "rb") as fh:
        r = requests.post("https://litterbox.catbox.moe/resources/internals/api.php",
                          data={"reqtype": "fileupload", "time": "72h"},
                          files={"fileToUpload": (video.name, fh, "video/mp4")}, timeout=900)
    url = r.text.strip()
    if r.status_code != 200 or not url.startswith("https://"):
        raise PublishError(f"litterbox: HTTP {r.status_code} {r.text[:120]}")
    return url


def _host_uguu(video: Path) -> str:
    with open(video, "rb") as fh:
        r = requests.post("https://uguu.se/upload", files={"files[]": (video.name, fh, "video/mp4")}, timeout=900)
    data = _check(r, "uguu")
    url = (data.get("files") or [{}])[0].get("url", "")
    if not url.startswith("https://"):
        raise PublishError(f"uguu: unexpected reply {str(data)[:120]}")
    return url


def _host_tmpfiles(video: Path) -> str:
    with open(video, "rb") as fh:
        r = requests.post("https://tmpfiles.org/api/v1/upload",
                          files={"file": (video.name, fh, "video/mp4")}, timeout=900)
    url = _check(r, "tmpfiles")["data"]["url"]
    return url.replace("tmpfiles.org/", "tmpfiles.org/dl/", 1).replace("http://", "https://", 1)


HOSTS = [_host_catbox, _host_litterbox, _host_uguu, _host_tmpfiles]


def _serves_mp4(url: str, size: int) -> str:
    """'' if the link serves our MP4 directly to anyone, else the reason it doesn't."""
    try:
        r = requests.get(url, headers={"Range": "bytes=0-31", "User-Agent": "facebookexternalhit/1.1"},
                         timeout=60, allow_redirects=True)
    except requests.RequestException as e:
        return f"link not reachable ({type(e).__name__})"
    if r.status_code not in (200, 206):
        return f"link returned HTTP {r.status_code}"
    head = r.content[:32]
    if head[4:8] != b"ftyp":
        return f"link doesn't serve the video directly (content-type {r.headers.get('content-type')})"
    total = r.headers.get("content-range", "").split("/")[-1]
    if total.isdigit() and int(total) != size:
        return f"link serves {total} bytes, expected {size}"
    return ""


def _containers_via_hosts(graph: str, user_id: str, token: str, video: Path, caption: str):
    """Yields container ids, one per host that works (the caller waits and moves on if processing fails)."""
    errors = []
    size = video.stat().st_size
    for host in HOSTS:
        name = host.__name__[6:]
        try:
            url = host(video)
        except (requests.RequestException, PublishError, KeyError, ValueError) as e:
            errors.append(f"{name}: {str(e)[:120]}")
            log(NAME, f"{name} upload failed ({str(e)[:100]}); trying the next host")
            continue
        problem = _serves_mp4(url, size)
        if problem:
            errors.append(f"{name}: {problem}")
            log(NAME, f"{name}: {problem}; trying the next host")
            continue
        log(NAME, f"video hosted temporarily via {name}")
        r = requests.post(f"{graph}/{user_id}/media", data={
            "media_type": "REELS", "video_url": url, "caption": caption,
            "share_to_feed": "true", "access_token": token}, timeout=120)
        try:
            yield name, _check(r, "create container (video_url)")["id"]
        except PublishError as e:
            errors.append(f"{name}: {str(e)[:160]}")
            log(NAME, f"{name}: Instagram refused the link ({str(e)[:100]})")
    raise PublishError("Every upload route failed: " + " | ".join(errors))


class ProcessingError(PublishError):
    pass


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
            raise ProcessingError(f"Instagram could not process the video: {data}")
        time.sleep(10)
    raise ProcessingError("Timed out waiting for Instagram to process the video")


def check_token(ctx: Context) -> tuple[bool, str]:
    """Quick check that the Instagram token still works (they expire after 60 days)."""
    ig = ctx.cfg["instagram"]
    host = os.getenv("IG_API_HOST") or ig["api_host"]
    try:
        r = requests.get(f"https://{host}/{ig['api_version']}/me",
                         params={"fields": "user_id,username", "access_token": os.getenv("IG_ACCESS_TOKEN", "")},
                         timeout=30)
    except requests.RequestException as e:
        return True, f"couldn't reach Instagram to check the token ({type(e).__name__}); continuing"
    if r.status_code == 200:
        return True, f"Instagram token OK (@{r.json().get('username', '?')})"
    return False, f"Instagram token problem: HTTP {r.status_code} {r.text[:200]}"


def publish_reel(ctx: Context, video: Path, caption: str) -> dict:
    ig = ctx.cfg["instagram"]
    token = os.getenv("IG_ACCESS_TOKEN", "")
    user_id = os.getenv("IG_USER_ID", "")
    if not token or not user_id:
        raise PublishError("IG_ACCESS_TOKEN and IG_USER_ID must be set")
    host = os.getenv("IG_API_HOST") or ig["api_host"]
    ver = ig["api_version"]
    graph = f"https://{host}/{ver}"

    container = None
    direct = host.endswith("facebook.com")  # direct upload only works with the Facebook-Login API
    if direct:
        try:
            container = _container_resumable(graph, ver, user_id, token, video, caption)
            log(NAME, f"container {container} — waiting for Instagram to process the video")
            _wait(graph, container, token)
        except (PublishError, requests.RequestException) as e:
            log(NAME, f"direct upload didn't work ({str(e)[:140]}); using public-link upload")
            container = None
    if container is None:
        for name, cid in _containers_via_hosts(graph, user_id, token, video, caption):
            log(NAME, f"container {cid} — waiting for Instagram to process the video")
            try:
                _wait(graph, cid, token)
                container = cid
                break
            except ProcessingError as e:
                log(NAME, f"Instagram couldn't process the video from {name} ({str(e)[:120]}); trying the next host")
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

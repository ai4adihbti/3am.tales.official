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
def _container_resumable(graph: str, ver: str, user_id: str, token: str, video: Path, caption: str,
                         extra: dict | None = None) -> str:
    r = requests.post(f"{graph}/{user_id}/media", data={
        "media_type": "REELS", "upload_type": "resumable", "caption": caption,
        "share_to_feed": "true", "access_token": token, **(extra or {})}, timeout=60)
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


def _containers_via_hosts(graph: str, user_id: str, token: str, video: Path, caption: str,
                          extra: dict | None = None):
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
            "share_to_feed": "true", "access_token": token, **(extra or {})}, timeout=120)
        if r.status_code >= 400 and extra and "cover_url" in extra:
            log(NAME, "Instagram refused the custom cover; posting with an automatic cover instead")
            extra = {k: v for k, v in extra.items() if k != "cover_url"}
            r = requests.post(f"{graph}/{user_id}/media", data={
                "media_type": "REELS", "video_url": url, "caption": caption,
                "share_to_feed": "true", "access_token": token, **extra}, timeout=120)
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


def followers(ctx: Context) -> int | None:
    """The account's follower count (None if it can't be read)."""
    token = os.getenv("IG_ACCESS_TOKEN", "")
    if not token or ctx.mock:
        return None
    ig = ctx.cfg["instagram"]
    host = os.getenv("IG_API_HOST") or ig["api_host"]
    try:
        r = requests.get(f"https://{host}/{ig['api_version']}/me",
                         params={"fields": "followers_count", "access_token": token}, timeout=30)
        n = r.json().get("followers_count") if r.status_code == 200 else None
        return int(n) if n is not None else None
    except (requests.RequestException, ValueError, TypeError):
        return None


def _host_cover(cover: Path) -> str:
    """Puts the cover image on a temporary public host (Instagram downloads it from there)."""
    for name, url_, data in (("catbox", "https://catbox.moe/user/api.php", {"reqtype": "fileupload"}),
                             ("litterbox", "https://litterbox.catbox.moe/resources/internals/api.php",
                              {"reqtype": "fileupload", "time": "72h"})):
        try:
            with open(cover, "rb") as fh:
                r = requests.post(url_, data=data, files={"fileToUpload": (cover.name, fh, "image/jpeg")}, timeout=120)
            url = r.text.strip()
            if r.status_code == 200 and url.startswith("https://"):
                head = requests.get(url, headers={"Range": "bytes=0-3"}, timeout=60).content[:3]
                if head == b"\xff\xd8\xff":
                    return url
        except requests.RequestException:
            pass
        log(NAME, f"cover upload via {name} failed; trying the next host")
    return ""


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


def _via_hosts(graph: str, user_id: str, token: str, video: Path, caption: str, extra: dict) -> str:
    for name, cid in _containers_via_hosts(graph, user_id, token, video, caption, extra):
        log(NAME, f"container {cid} — waiting for Instagram to process the video")
        try:
            _wait(graph, cid, token)
            return cid
        except ProcessingError as e:
            log(NAME, f"Instagram couldn't process the video from {name} ({str(e)[:120]}); trying the next host")
    raise PublishError("no host worked")


def publish_reel(ctx: Context, video: Path, caption: str, cover: Path | None = None) -> dict:
    ig = ctx.cfg["instagram"]
    token = os.getenv("IG_ACCESS_TOKEN", "")
    user_id = os.getenv("IG_USER_ID", "")
    if not token or not user_id:
        raise PublishError("IG_ACCESS_TOKEN and IG_USER_ID must be set")
    host = os.getenv("IG_API_HOST") or ig["api_host"]
    ver = ig["api_version"]
    graph = f"https://{host}/{ver}"

    extra = {}
    if cover and cover.exists():
        cover_url = _host_cover(cover)
        if cover_url:
            extra["cover_url"] = cover_url
            log(NAME, "custom cover attached")
        else:
            extra["thumb_offset"] = str(int(ctx.notes.get("cover_offset_ms", 1500)))
    container = None
    direct = host.endswith("facebook.com")  # direct upload only works with the Facebook-Login API
    if direct:
        try:
            container = _container_resumable(graph, ver, user_id, token, video, caption, extra)
            log(NAME, f"container {container} — waiting for Instagram to process the video")
            _wait(graph, container, token)
        except (PublishError, requests.RequestException) as e:
            log(NAME, f"direct upload didn't work ({str(e)[:140]}); using public-link upload")
            container = None
    if container is None:
        try:
            container = _via_hosts(graph, user_id, token, video, caption, extra)
        except PublishError as e:
            if not extra:
                raise
            log(NAME, f"posting with the custom cover failed ({str(e)[:120]}); retrying without it")
            container = _via_hosts(graph, user_id, token, video, caption, {})
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

"""Civitai lookup by file hash. Only JSON metadata is fetched (sizes, prompts, flags);
example images are never downloaded. Hashes and responses are cached under output/cache/."""
import json
import math
import time
from pathlib import Path

import requests

from .hashing import file_sha256
from .store import read_json, write_json_atomic

API = "https://civitai.com/api/v1/model-versions/by-hash/{}"

# Standard SDXL / Pony V6 training resolutions (~1 megapixel).
BUCKETS = [(1024, 1024), (1152, 896), (896, 1152), (1216, 832), (832, 1216),
           (1344, 768), (768, 1344), (1536, 640), (640, 1536)]

BUCKET_NAMES = {f"{w}x{h}" for w, h in BUCKETS}

_last_request = 0.0


def nearest_bucket(width: int, height: int) -> tuple[int, int]:
    """Scale any resolution to the Pony bucket with the closest aspect ratio."""
    ratio = math.log(width / height)
    return min(BUCKETS, key=lambda b: abs(math.log(b[0] / b[1]) - ratio))


def _cache_dir(cfg) -> Path:
    d = cfg.assert_writable(cfg.output_dir / "cache")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fetch(sha: str, cfg) -> dict | None:
    """Raw by-hash response, cached. None means Civitai doesn't know this file."""
    cache_file = _cache_dir(cfg) / "civitai" / f"{sha}.json"
    data = read_json(cache_file) if cache_file.exists() else None
    if isinstance(data, dict):
        return None if data.get("not_found") else data
    global _last_request
    delay = cfg.civitai.get("request_delay_s", 0.5) - (time.time() - _last_request)
    if delay > 0:
        time.sleep(delay)
    _last_request = time.time()
    r = requests.get(API.format(sha), timeout=30)
    if r.status_code == 404:
        data = {"not_found": True}
    elif r.status_code == 200:
        data = r.json()
    else:
        raise requests.HTTPError(f"Civitai HTTP {r.status_code}: {r.text[:200]}")
    write_json_atomic(cache_file, data, indent=None)
    return None if data.get("not_found") else data


def lookup(path: Path, cfg) -> dict | None:
    """Summarized Civitai info for a LoRA file, or None if unavailable."""
    sha = file_sha256(path, cfg)
    data = _fetch(sha, cfg)
    if not data:
        return None
    images = []
    for img in data.get("images") or []:
        w, h = img.get("width"), img.get("height")
        if not (w and h):
            continue
        meta = img.get("meta") or {}
        images.append({"width": w, "height": h, "bucket": list(nearest_bucket(w, h)),
                       "nsfw_level": img.get("nsfwLevel"), "minor": bool(img.get("minor")),
                       "prompt": (meta.get("prompt") or "")[:600]})
    model = data.get("model") or {}
    return {"sha256": sha, "model_id": data.get("modelId"), "version_id": data.get("id"),
            "model_name": model.get("name"), "version_name": data.get("name"),
            "base_model": data.get("baseModel"), "trained_words": data.get("trainedWords") or [],
            "images": images}


def bucket_plan(info: dict | None, count: int) -> list[tuple[int, int]]:
    """Resolutions for `count` probes, split in proportion to how often each bucket
    appears in the Civitai examples (largest-remainder rounding, most common first)."""
    freq: dict[tuple[int, int], int] = {}
    for img in (info or {}).get("images", []):
        b = tuple(img["bucket"])
        freq[b] = freq.get(b, 0) + 1
    if not freq:
        return [(1024, 1024)] * count
    total = sum(freq.values())
    ranked = sorted(freq.items(), key=lambda kv: -kv[1])
    shares = {b: n * count / total for b, n in ranked}
    alloc = {b: int(s) for b, s in shares.items()}
    for b, _ in sorted(shares.items(), key=lambda kv: -(kv[1] - int(kv[1])))[: count - sum(alloc.values())]:
        alloc[b] += 1
    # Interleave so every resolution is covered early (matters for baselines, which copy the first probes).
    plan, queues = [], [[b] * alloc[b] for b, _ in ranked if alloc[b]]
    while any(queues):
        for q in queues:
            if q:
                plan.append(q.pop())
    return plan

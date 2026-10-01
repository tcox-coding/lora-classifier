import hashlib
import json
import re
import struct
from pathlib import Path

from . import civitai
from .safety import is_sexual_tag, minor_terms_in

# Tags too generic to act as triggers (still usable as context tags).
GENERIC_TAGS = {
    "1girl", "1boy", "2girls", "2boys", "3girls", "multiple girls", "multiple boys", "solo", "solo focus",
    "breasts", "large breasts", "medium breasts", "small breasts", "huge breasts", "nipples", "nude",
    "completely nude", "looking at viewer", "long hair", "short hair", "blush", "smile", "open mouth",
    "simple background", "white background", "hetero", "penis", "pussy", "sex", "navel", "thighs",
    "closed eyes", "black hair", "brown hair", "blonde hair", "blue eyes", "brown eyes", "indoors", "outdoors",
}
# Tags describing the training images rather than the concept; never put in prompts.
NOISE_TAGS = {
    "censored", "mosaic censoring", "bar censor", "uncensored", "watermark", "signature", "artist name",
    "text", "english text", "web address", "patreon username", "twitter username", "speech bubble",
    "commentary request", "highres", "absurdres", "lowres", "comic", "translated", "dated",
}


def read_header(path: Path) -> dict:
    with open(path, "rb") as fh:
        (length,) = struct.unpack("<Q", fh.read(8))
        if length > 100_000_000:
            raise ValueError("implausible safetensors header size")
        header = json.loads(fh.read(length))
    return header.get("__metadata__") or {}


def lora_id(rel_path: str) -> str:
    stem = Path(rel_path).stem
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", stem).strip("_")[:48] or "lora"
    return f"{slug}_{hashlib.sha1(rel_path.lower().encode()).hexdigest()[:8]}"


def _json_field(meta: dict, key: str):
    try:
        return json.loads(meta[key]) if key in meta else None
    except (TypeError, json.JSONDecodeError):
        return None


def _rgthree_info(path: Path) -> dict:
    info_path = path.with_name(path.name + ".rgthree-info.json")
    if not info_path.exists():
        return {}
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    words = []
    for w in info.get("trainedWords", []) or []:
        words.append(w.get("word") if isinstance(w, dict) else str(w))
    prompts = [img.get("positive", "")[:400] for img in (info.get("images") or [])[:3] if img.get("positive")]
    return {"trained_words": [w for w in words if w], "civitai_example_prompts": prompts,
            "civitai_name": info.get("name"), "civitai_description": (info.get("description") or "")[:1500]}


def extract(path: Path, cfg) -> dict:
    path = Path(path)
    rel_in_source = path.relative_to(cfg.source_dir).as_posix()
    comfy_name = str(path.relative_to(cfg.loras_root)).replace("/", "\\")
    stat = path.stat()
    meta = {
        "id": lora_id(rel_in_source),
        "filename": path.name,
        "source_path": str(path),
        "source_rel": rel_in_source,
        "source_folder": str(Path(rel_in_source).parent.as_posix()),
        "comfy_name": comfy_name,
        "size_mb": round(stat.st_size / 2**20, 1),
        "mtime": stat.st_mtime,
    }
    try:
        raw = read_header(path)
    except Exception as e:  # noqa: BLE001
        meta["header_error"] = str(e)
        raw = {}

    meta["title"] = raw.get("modelspec.title") or raw.get("ss_output_name")
    meta["base_model"] = raw.get("ss_base_model_version") or raw.get("modelspec.architecture")
    meta["trained_on"] = raw.get("ss_sd_model_name")
    meta["network_dim"] = raw.get("ss_network_dim")
    meta["network_module"] = raw.get("ss_network_module")
    if raw.get("modelspec.description"):
        meta["embedded_description"] = raw["modelspec.description"][:1500]

    # Merge tag frequencies across dataset folders.
    freq: dict[str, int] = {}
    for folder_tags in (_json_field(raw, "ss_tag_frequency") or {}).values():
        for tag, count in folder_tags.items():
            tag = tag.replace("﻿", "").strip()
            if tag:
                freq[tag] = freq.get(tag, 0) + int(count)
    img_count = sum(int(d.get("img_count", 0)) for d in (_json_field(raw, "ss_dataset_dirs") or {}).values())
    img_count = img_count or (max(freq.values()) if freq else 0)
    meta["training_images"] = img_count
    meta["tag_frequency"] = dict(sorted(freq.items(), key=lambda kv: -kv[1])[:60])
    # Safety checks look at every caption tag, not just the most common ones.
    meta["minor_tag_hits"] = {t: n for t, n in freq.items() if minor_terms_in(t)}
    meta["sexual_tag_hits"] = {t: n for t, n in freq.items() if is_sexual_tag(t)}

    meta.update(_rgthree_info(path))
    meta.setdefault("trained_words", [])
    if raw.get("modelspec.trigger_phrase"):
        meta["trained_words"].insert(0, raw["modelspec.trigger_phrase"])

    meta["triggers"], meta["top_tags"] = _pick_tags(meta, freq, img_count)
    return meta


PONY_META_RE = re.compile(r"^(score_\d(_up)?|source_\w+|rating_\w+)$", re.I)


def _usable(tag: str) -> bool:
    return (0 < len(tag) <= 60 and tag.lower() not in NOISE_TAGS
            and not PONY_META_RE.match(tag) and not minor_terms_in(tag))


def _pick_tags(meta: dict, freq: dict, img_count: int) -> tuple[list[str], list[str]]:
    triggers = [w.replace("﻿", "").strip() for w in meta["trained_words"] if w.strip() and _usable(w)][:3]
    ranked = [t for t, _ in sorted(freq.items(), key=lambda kv: -kv[1]) if _usable(t)]
    if not triggers and img_count:
        # Tags present on (nearly) every training image are almost always the activation tag.
        triggers = [t for t in ranked if t.lower() not in GENERIC_TAGS and freq[t] >= 0.75 * img_count][:3]
        if not triggers and ranked:
            top = next((t for t in ranked if t.lower() not in GENERIC_TAGS), None)
            if top and freq[top] >= 0.4 * img_count:
                triggers = [top]
    top_tags = [t for t in ranked if t not in triggers][:10]
    return triggers, top_tags


def enrich(meta: dict, cfg) -> dict:
    """Add Civitai info (if enabled and found) and fold its trigger words into ours."""
    if not cfg.civitai.get("enabled") or "civitai" in meta:
        return meta
    info = civitai.lookup(Path(meta["source_path"]), cfg)
    meta["civitai"] = info
    if info and info["trained_words"]:
        words = [w for w in info["trained_words"] if w not in meta["trained_words"]]
        meta["trained_words"] = meta["trained_words"] + words
        meta["triggers"], meta["top_tags"] = _pick_tags(meta, meta["tag_frequency"], meta["training_images"])
    return meta


def scan(cfg) -> list[Path]:
    return sorted(cfg.source_dir.rglob("*.safetensors"), key=lambda p: str(p).lower())

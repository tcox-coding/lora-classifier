"""Blocklist of LoRA files by SHA-256.

Blocking a LoRA records its hash in blocklist/<sha256>.json, permanently deletes the
.safetensors file (and this tool's hardlink, images and description of it) and logs it in
blocklist/blocklist.log. Before every run, the scan folder is swept for files with a blocked
hash (e.g. the same LoRA downloaded again); matches are deleted and the log is updated.

A file is only ever deleted if it is a .safetensors file under the LoRA folders AND a fresh
hash of its contents matches a blocked entry.
"""
import json
import shutil
from pathlib import Path

from . import hashing, reviews
from .store import append_line, load_state, log, read_json, save_state, write_json_atomic

LOG_NAME = "blocklist.log"


def _entry_path(cfg, sha: str) -> Path:
    return cfg.assert_writable(cfg.blocklist_dir / f"{sha}.json")


def entries(cfg) -> list[dict]:
    if not cfg.blocklist_dir.exists():
        return []
    out = []
    for p in sorted(cfg.blocklist_dir.glob("*.json")):
        entry = read_json(p, set_aside=False)
        if not isinstance(entry, dict) or not entry.get("sha256"):
            # Damaged entry: the hash is the file name, so the LoRA stays blocked regardless.
            entry = {"sha256": p.stem, "size": None, "filename": "(damaged blocklist entry)", "rejections": []}
        out.append(entry)
    return out


def is_blocked(cfg, sha: str | None) -> bool:
    return bool(sha) and (cfg.blocklist_dir / f"{sha}.json").exists()


def _write_log(cfg, action: str, sha: str, path: str, note: str = ""):
    p = cfg.assert_writable(cfg.blocklist_dir / LOG_NAME)
    append_line(p, f"{reviews.now()}  {action:<14} {sha}  {path}{('  ' + note) if note else ''}")


def read_log(cfg) -> str:
    p = cfg.blocklist_dir / LOG_NAME
    return p.read_text(encoding="utf-8") if p.exists() else ""


def delete_verified(cfg, path: Path, sha: str) -> bool:
    """Delete a LoRA file only if it's in the LoRA folders and its contents hash to `sha`."""
    target = cfg.assert_deletable_lora(path)
    if not target.exists():
        return False
    actual = hashing.sha256_of(target)
    if actual != sha:
        raise PermissionError(f"refusing to delete {target}: its hash {actual[:12]}… doesn't match {sha[:12]}…")
    target.unlink()
    return True


def _clear_outputs(cfg, state: dict | None, lora_id: str):
    """Remove everything this tool made for a LoRA: link, description, images, test renders."""
    from . import organize
    if state:
        for key in ("lora", "description_json", "description_md"):
            organize._unlink(cfg, (state.get("placement") or {}).get(key))
    for d in (cfg.examples_dir / lora_id, cfg.tests_dir / lora_id):
        if d.exists():
            shutil.rmtree(cfg.assert_writable(d))


def _mark_state_blocked(cfg, state: dict, sha: str, reason: str):
    state["status"] = "blocked"
    state["stages"] = {}
    state["placement"] = None
    state.setdefault("safety", {})["blocked"] = {"sha256": sha, "blocked_at": reviews.now(), "reason": reason}
    save_state(cfg, state)


def block(cfg, meta: dict, reason: str, taxonomy=None) -> dict:
    """Blocklist a LoRA and permanently delete its file. Returns the blocklist entry."""
    from . import organize
    source = Path(meta["source_path"])
    cfg.assert_deletable_lora(source)
    # Verify the file before touching anything, so a mismatch leaves everything as it was.
    if source.exists():
        sha = hashing.sha256_of(source)
        if meta.get("sha256") and meta["sha256"] != sha:
            raise PermissionError(f"{source} has changed since it was reviewed (hash mismatch); refusing to block")
    else:
        sha = meta.get("sha256")
    if not sha:
        raise FileNotFoundError(f"cannot block {source}: file is missing and no hash is recorded")

    entry_path = _entry_path(cfg, sha)
    entry_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"sha256": sha, "size": source.stat().st_size if source.exists() else meta.get("size_bytes"),
             "filename": meta["filename"], "source_path": str(source), "lora_id": meta["id"],
             "blocked_at": reviews.now(), "reason": reason,
             "civitai": (meta.get("civitai") or {}).get("model_name"), "rejections": []}
    old = read_json(entry_path, set_aside=False) if entry_path.exists() else None
    if isinstance(old, dict):  # already blocked: keep the original record and history
        entry = {**old, "reason_again": reason}
    write_json_atomic(entry_path, entry)

    state = load_state(cfg, meta["id"])
    if state and taxonomy and (state.get("placement") or {}).get("counted"):
        taxonomy.remove(state["placement"]["category_path"])
    _clear_outputs(cfg, state, meta["id"])
    deleted = delete_verified(cfg, source, sha)
    _write_log(cfg, "BLOCKED", sha, str(source), f"deleted={deleted}  reason: {reason}")
    reviews.revoke(cfg, sha, meta, "blocked")
    reviews.log_event(cfg, sha, meta, "blocked", reason=reason, file_deleted=deleted)
    _mark_state_blocked(cfg, state or {"meta": meta, "stages": {}, "safety": {}}, sha, reason)
    organize.write_index(cfg)
    log(f"BLOCKED {meta['source_rel']} (file deleted: {deleted})")
    return entry


def enforce(cfg, dry_run: bool = False) -> list[dict]:
    """Find (and unless dry_run, delete) files in the scan folder whose hash is blocked."""
    blocked = {e["sha256"]: e for e in entries(cfg) if e.get("sha256")}
    if not blocked:
        return []
    sizes = {e.get("size") for e in blocked.values()}
    size_filter = None not in sizes  # only hash files whose size matches a blocked file
    found = []
    root = cfg.blocklist_scan_dir
    for f in sorted(root.rglob("*.safetensors")) if root.exists() else []:
        try:
            if size_filter and f.stat().st_size not in sizes:
                continue
            sha = hashing.file_sha256(f, cfg)
        except OSError:
            continue
        if sha not in blocked:
            continue
        match = {"path": str(f), "sha256": sha, "filename": blocked[sha].get("filename"), "deleted": False}
        if not dry_run:
            match["deleted"] = delete_verified(cfg, f, sha)
            _write_log(cfg, "AUTO-REJECTED", sha, str(f), f"deleted={match['deleted']}  "
                       f"(blocked on {blocked[sha].get('blocked_at')} as {blocked[sha].get('filename')})")
            entry = blocked[sha]
            entry.setdefault("rejections", []).append({"time": reviews.now(), "path": str(f),
                                                       "deleted": match["deleted"]})
            write_json_atomic(_entry_path(cfg, sha), entry)
            _mark_rejected_state(cfg, f, sha)
            log(f"AUTO-REJECTED blocked LoRA {f} (deleted: {match['deleted']})")
        found.append(match)
    return found


def _mark_rejected_state(cfg, path: Path, sha: str):
    """If the rejected file sits in the source folder and has a state, mark it blocked."""
    from . import metadata
    try:
        rel = path.resolve().relative_to(cfg.source_dir).as_posix()
    except ValueError:
        return
    lora_id = metadata.lora_id(rel)
    state = load_state(cfg, lora_id)
    if state:
        _clear_outputs(cfg, state, lora_id)
        _mark_state_blocked(cfg, state, sha, "auto-rejected: hash is on the blocklist")

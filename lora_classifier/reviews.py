"""Manual review records, keyed by the LoRA file's SHA-256 so an approval never carries over
to a different file. approvals.json and quarantine.json hold the current decisions;
review_log.jsonl is an append-only history of every review action, test render and automatic
revocation. If either decision file is damaged (e.g. by a crash), it is rebuilt from the log,
so a quarantine can never be silently lost."""
import json
import threading
import time
from pathlib import Path

from .store import append_line, log, read_json, write_json_atomic

_lock = threading.Lock()


def _approvals_file(cfg) -> Path:
    return cfg.assert_writable(cfg.review_dir / "approvals.json")


def _log_file(cfg) -> Path:
    return cfg.assert_writable(cfg.review_dir / "review_log.jsonl")


def _replay(cfg) -> tuple[dict, dict]:
    """Rebuild current approvals and quarantines from the append-only log."""
    approvals, quarantined = {}, {}
    for e in _events(cfg):
        sha, ev = e.get("sha256"), e.get("event", "")
        if not sha:
            continue
        if ev == "approved":
            approvals[sha] = {"approved_at": e["time"], "notes": e.get("notes", ""), "lora_id": e.get("lora_id"),
                              "file": e.get("file"), "sexual_at_approval": e.get("sexual")}
        elif ev.startswith("approval revoked") or ev in ("quarantined", "blocked"):
            approvals.pop(sha, None)
        if ev == "quarantined":
            quarantined[sha] = {"quarantined_at": e["time"], "notes": e.get("notes", ""),
                                "lora_id": e.get("lora_id"), "file": e.get("file")}
        elif ev == "released from quarantine":
            quarantined.pop(sha, None)
    return approvals, quarantined


_decision_cache: dict[str, tuple[int, dict]] = {}  # path -> (mtime_ns, parsed contents)


def _load_decisions(cfg, path: Path, which: int) -> dict:
    try:
        mtime = path.stat().st_mtime_ns
    except FileNotFoundError:
        return {}
    hit = _decision_cache.get(str(path))
    if hit and hit[0] == mtime:
        return dict(hit[1])
    data = read_json(path)  # a damaged file is moved aside and comes back as None
    if isinstance(data, dict):
        _decision_cache[str(path)] = (mtime, data)
        return dict(data)
    rebuilt = _replay(cfg)[which]
    write_json_atomic(path, rebuilt)
    log(f"WARNING: {path.name} was damaged; rebuilt {len(rebuilt)} record(s) from the review log")
    return rebuilt


def _load(cfg) -> dict:
    return _load_decisions(cfg, _approvals_file(cfg), 0)


def _save(cfg, data: dict):
    write_json_atomic(_approvals_file(cfg), data)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log_event(cfg, sha: str, meta: dict | None, event: str, **details):
    entry = {"time": now(), "sha256": sha, "event": event,
             "lora_id": (meta or {}).get("id"), "file": (meta or {}).get("source_rel"), **details}
    p = _log_file(cfg)
    with _lock:
        append_line(p, json.dumps(entry, ensure_ascii=False))


def _events(cfg) -> list[dict]:
    """Every readable log entry. A line cut short by a crash is skipped."""
    p = _log_file(cfg)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(e, dict):
            out.append(e)
    return out


def history(cfg, sha: str) -> list[dict]:
    return [e for e in _events(cfg) if e.get("sha256") == sha]


def approval(cfg, sha: str | None) -> dict | None:
    return _load(cfg).get(sha) if sha else None


def _quarantine_file(cfg) -> Path:
    return cfg.assert_writable(cfg.review_dir / "quarantine.json")


def _load_quarantine(cfg) -> dict:
    return _load_decisions(cfg, _quarantine_file(cfg), 1)


def _save_quarantine(cfg, data: dict):
    write_json_atomic(_quarantine_file(cfg), data)


def quarantine_record(cfg, sha: str | None) -> dict | None:
    return _load_quarantine(cfg).get(sha) if sha else None


def quarantine(cfg, sha: str, meta: dict, notes: str):
    """Manual review decision: never render this file. Also revokes any approval."""
    with _lock:
        data = _load_quarantine(cfg)
        data[sha] = {"quarantined_at": now(), "notes": notes, "lora_id": meta["id"], "file": meta["source_rel"]}
        _save_quarantine(cfg, data)
    revoke(cfg, sha, meta, "quarantined")
    log_event(cfg, sha, meta, "quarantined", notes=notes)


def release(cfg, sha: str, meta: dict, notes: str) -> bool:
    with _lock:
        data = _load_quarantine(cfg)
        existed = data.pop(sha, None) is not None
        _save_quarantine(cfg, data)
    if existed:
        log_event(cfg, sha, meta, "released from quarantine", notes=notes)
    return existed


def approve(cfg, sha: str, meta: dict, notes: str, assessment: dict):
    """Record a manual review approval. Refuses quarantined LoRAs (release them first)."""
    if quarantine_record(cfg, sha):
        raise PermissionError("this LoRA is quarantined; release it from quarantine before approving")
    record = {"approved_at": now(), "notes": notes, "lora_id": meta["id"], "file": meta["source_rel"],
              "sexual_at_approval": assessment["sexual"]}
    with _lock:
        data = _load(cfg)
        data[sha] = record
        _save(cfg, data)
    log_event(cfg, sha, meta, "approved", notes=notes, tier=assessment.get("tier"),
              sexual=assessment["sexual"])


def revoke(cfg, sha: str, meta: dict | None, reason: str, automatic: bool = False):
    with _lock:
        data = _load(cfg)
        existed = data.pop(sha, None) is not None
        _save(cfg, data)
    if existed:
        log_event(cfg, sha, meta, "approval revoked automatically" if automatic else "approval revoked",
                  reason=reason)
    return existed

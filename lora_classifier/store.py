"""Per-LoRA state files, crash-safe JSON writing and the shared log."""
import json
import os
import time
from pathlib import Path

_log_sink = None


def set_log_sink(fn):
    """Send log lines to fn (e.g. the UI) instead of stdout. None restores printing."""
    global _log_sink
    _log_sink = fn


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    if _log_sink:
        _log_sink(line)
    else:
        print(line, flush=True)


# ---------------------------------------------------------------- crash-safe JSON

def write_json_atomic(path: Path, data, indent: int | None = 1):
    """Write JSON so a crash or power loss leaves either the old file or the new one, never a
    truncated mix: write a temp file, force it to disk, then swap it in."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=indent)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def append_line(path: Path, line: str):
    """Append one line durably. If a crash cut the previous line short, start on a fresh line
    so the new entry isn't glued onto the broken one and lost with it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_newline = False
    if path.exists() and path.stat().st_size:
        with open(path, "rb") as fh:
            fh.seek(-1, os.SEEK_END)
            needs_newline = fh.read(1) != b"\n"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(("\n" if needs_newline else "") + line.rstrip("\n") + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def set_aside_corrupt(path: Path, why: str) -> Path:
    """Move an unreadable file out of the way (kept for inspection) and log it."""
    path = Path(path)
    dest = path.with_name(f"{path.name}.corrupt-{time.strftime('%Y%m%d-%H%M%S')}")
    os.replace(path, dest)
    log(f"WARNING: {path.name} was unreadable ({why}); moved to {dest.name}")
    return dest


def read_json(path: Path, default=None, set_aside: bool = True):
    """Read JSON. A missing file returns `default`; an unreadable one (e.g. truncated by a crash)
    is moved aside and also returns `default`."""
    path = Path(path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as e:
        if set_aside:
            set_aside_corrupt(path, type(e).__name__)
        return default


# ---------------------------------------------------------------- per-LoRA state

def state_path(cfg, lora_id: str) -> Path:
    return cfg.state_dir / f"{lora_id}.json"


def normalize(state: dict) -> dict:
    """Older records used "high_risk"; it's now "caution" (set aside, awaiting manual review)."""
    if state.get("status") == "high_risk":
        state["status"] = "caution"
    return state


def _valid(state) -> bool:
    return isinstance(state, dict) and isinstance(state.get("meta"), dict) and "id" in state["meta"]


def load_state(cfg, lora_id: str) -> dict | None:
    """A LoRA's record, or None if it has none. A damaged record is moved aside and treated as
    missing, so the LoRA is simply processed again."""
    p = state_path(cfg, lora_id)
    state = read_json(p)
    if state is not None and not _valid(state):
        set_aside_corrupt(p, "missing fields")
        return None
    return normalize(state) if state is not None else None


def all_states(cfg) -> list[dict]:
    if not cfg.state_dir.exists():
        return []
    out = []
    for p in sorted(cfg.state_dir.glob("*.json")):
        state = read_json(p)
        if state is None:
            continue
        if not _valid(state):
            set_aside_corrupt(p, "missing fields")
            continue
        out.append(normalize(state))
    return out


def save_state(cfg, state: dict):
    write_json_atomic(cfg.assert_writable(state_path(cfg, state["meta"]["id"])), state)

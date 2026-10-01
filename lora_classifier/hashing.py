"""File identity. The full-file SHA-256 identifies a LoRA for reviews, the blocklist and
Civitai (which calls it AutoV2). Hashes are cached by path + size + mtime, on disk and in memory."""
import hashlib
import json
import threading
from pathlib import Path

_lock = threading.Lock()
_memory: dict[str, dict] = {}  # cache file path -> its contents, loaded once per process


def sha256_of(path: Path) -> str:
    """Hash the file now, ignoring the cache (used before anything irreversible)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache(cache_file: Path) -> dict:
    key = str(cache_file)
    if key not in _memory:
        try:
            _memory[key] = json.loads(cache_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            _memory[key] = {}
    return _memory[key]


_cache_files: dict[str, Path] = {}


def _cache_file(cfg) -> Path:
    key = str(cfg.output_dir)
    if key not in _cache_files:
        cache_dir = cfg.assert_writable(cfg.output_dir / "cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        _cache_files[key] = cache_dir / "hashes.json"
    return _cache_files[key]


def file_sha256(path: Path, cfg) -> str:
    path = Path(path)
    cache_file = _cache_file(cfg)
    stat = path.stat()
    key = f"{path}|{stat.st_size}|{int(stat.st_mtime)}"
    with _lock:
        cache = _cache(cache_file)
        if key in cache:
            return cache[key]
    digest = sha256_of(path)
    with _lock:
        cache = _cache(cache_file)
        cache[key] = digest
        from .store import write_json_atomic
        write_json_atomic(cache_file, cache, indent=0)
    return digest

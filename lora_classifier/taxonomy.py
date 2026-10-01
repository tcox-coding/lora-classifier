"""Dynamic category tree. Seeded from the existing hand-sorted folders, grown by the model."""
import re

from . import store

from .config import SAFETY_FOLDERS

MAX_DEPTH = 7
MAX_SEGMENT = 40


def clean_path(segments) -> list[str]:
    if isinstance(segments, str):
        segments = re.split(r"[\\/]", segments)
    out = []
    for seg in segments or []:
        seg = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", str(seg)).lower()
        seg = re.sub(r"\s+", " ", seg).strip(" .")[:MAX_SEGMENT].strip(" .")
        if seg and seg not in SAFETY_FOLDERS:  # only the safety pipeline may assign these folders
            out.append(seg)
    return out[:MAX_DEPTH] or ["uncategorized"]


class Taxonomy:
    def __init__(self, cfg):
        self.cfg = cfg
        self.file = cfg.taxonomy_file
        self.paths: dict[str, int] = {}
        existed = self.file.exists()
        data = store.read_json(self.file)
        if isinstance(data, dict):
            self.paths = data
        else:
            self._seed()
            if existed:  # damaged: rebuild the counts from the records
                for s in store.all_states(cfg):
                    pl = s.get("placement") or {}
                    if pl.get("counted") and pl.get("category_path"):
                        key = "/".join(pl["category_path"])
                        self.paths[key] = self.paths.get(key, 0) + 1
                self.save()
                store.log("taxonomy.json was damaged and has been rebuilt from the saved records")

    def _seed(self):
        # Existing folders that hold LoRAs are the starting vocabulary (count 0 = not yet confirmed).
        for f in self.cfg.source_dir.rglob("*.safetensors"):
            rel = f.parent.relative_to(self.cfg.source_dir).as_posix()
            if rel != ".":
                self.paths.setdefault("/".join(clean_path(rel)), 0)
        self.save()

    def add(self, segments: list[str]):
        key = "/".join(segments)
        self.paths[key] = self.paths.get(key, 0) + 1
        self.save()

    def remove(self, segments: list[str]):
        key = "/".join(segments)
        if self.paths.get(key, 0) > 0:
            self.paths[key] -= 1
            self.save()

    def save(self):
        store.write_json_atomic(self.file, dict(sorted(self.paths.items())))

    def listing(self) -> str:
        lines = []
        for path, n in sorted(self.paths.items()):
            lines.append(f"{path}  ({n} classified)" if n else path)
        return "\n".join(lines)

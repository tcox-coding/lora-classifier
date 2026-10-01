import os
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Top-level categories for LoRAs set aside by the safety checks.
CAUTION_FOLDER = "caution loras"          # set aside automatically, awaiting manual review
QUARANTINE_FOLDER = "quarantined loras"   # quarantined by manual review
HIGH_RISK_FOLDER = "high risk loras"      # name used by older versions
SAFETY_FOLDERS = (CAUTION_FOLDER, QUARANTINE_FOLDER, HIGH_RISK_FOLDER)


def _read_dotenv(path: Path) -> dict:
    values = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _resolve(base: Path, value: str) -> Path:
    p = Path(value)
    return (p if p.is_absolute() else base / p).resolve()


class Config:
    def __init__(self, path: str | Path | None = None):
        path = Path(path or ROOT / "config.toml").resolve()
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        base = path.parent

        paths = raw["paths"]
        self.loras_root = Path(paths["loras_root"]).resolve()
        self.source_subdir = paths["source_subdir"]
        self.source_dir = (self.loras_root / self.source_subdir).resolve()
        self.output_dir = _resolve(base, paths["output_dir"])
        # Kept outside output/ so rebuilding output can never lose reviews or the blocklist.
        self.review_dir = _resolve(base, paths.get("review_dir", "review"))
        self.blocklist_dir = _resolve(base, paths.get("blocklist_dir", "blocklist"))

        self.deepinfra = raw["deepinfra"]
        self.comfy = raw["comfy"]
        self.probe = raw["probe"]
        self.examples = raw["examples"]
        self.civitai = raw.get("civitai", {"enabled": False})
        self.link_mode = raw["organize"]["link_mode"]
        blocklist = raw.get("blocklist", {})
        # Folder swept for blocked hashes before every run (defaults to the source folder).
        self.blocklist_scan_dir = _resolve(self.loras_root, blocklist.get("scan_dir", str(self.source_dir)))

        self.api_key = os.environ.get("DEEPINFRA_API_KEY") or _read_dotenv(base / ".env").get("DEEPINFRA_API_KEY")

        # Output layout.
        self.loras_out = self.output_dir / "loras" / self.source_subdir
        self.catalog_dir = self.output_dir / "catalog"
        self.descriptions_dir = self.catalog_dir / "descriptions"
        self.examples_dir = self.catalog_dir / "examples"
        self.tests_dir = self.catalog_dir / "tests"
        self.taxonomy_file = self.catalog_dir / "taxonomy.json"
        self.state_dir = self.output_dir / "state"

        # The source library is read-only apart from explicit blocklist deletions: refuse any
        # layout that would put the tool's own files inside it.
        for d in (self.output_dir, self.review_dir, self.blocklist_dir):
            if d == self.source_dir or self.source_dir in d.parents:
                raise ValueError(f"{d} must not be inside the source LoRA folder")

    def assert_writable(self, target: Path) -> Path:
        """Every normal write goes through here: only the tool's own folders are allowed."""
        target = Path(target).resolve()
        for root in (self.output_dir, self.review_dir, self.blocklist_dir):
            if target == root or root in target.parents:
                return target
        raise PermissionError(f"refusing to write outside the tool's folders: {target}")

    def assert_deletable_lora(self, target: Path) -> Path:
        """The only way a file outside the tool's folders may be deleted: a .safetensors file
        under the LoRA root or the blocklist scan folder. Callers must also verify its hash."""
        target = Path(target).resolve()
        roots = (self.loras_root, self.blocklist_scan_dir)
        if target.suffix.lower() != ".safetensors" or not any(r in target.parents for r in roots):
            raise PermissionError(f"refusing to delete {target}: not a .safetensors file under the LoRA folders")
        return target

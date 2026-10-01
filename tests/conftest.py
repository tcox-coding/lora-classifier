"""Test fixtures: a throwaway LoRA library in a temp folder, a fake ComfyUI and a fake DeepInfra.
Nothing here touches the real library, the real output folder or the network."""
import io
import json
import struct
import textwrap
from pathlib import Path

import pytest
from PIL import Image

from lora_classifier import comfy as comfy_mod
from lora_classifier import config as config_mod
from lora_classifier import safety, store
from lora_classifier.config import Config

REAL_PROJECT = config_mod.ROOT


# ---------------------------------------------------------------- safety net

@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("tests must not use the network")
    import requests
    monkeypatch.setattr(requests, "get", refuse)
    monkeypatch.setattr(requests, "post", refuse)
    monkeypatch.setattr(requests.Session, "post", refuse)


@pytest.fixture(autouse=True)
def quiet_log():
    lines = []
    store.set_log_sink(lines.append)
    yield lines
    store.set_log_sink(None)


# ---------------------------------------------------------------- fake library

def write_lora(path: Path, tags: dict[str, int] | None = None, images: int = 20, title: str | None = None,
               payload: bytes | None = None):
    """Minimal valid .safetensors file with kohya-style training metadata."""
    meta = {"ss_base_model_version": "sdxl_base_v1-0", "ss_output_name": title or path.stem}
    if tags is not None:
        meta["ss_tag_frequency"] = json.dumps({"1_data": tags})
        meta["ss_dataset_dirs"] = json.dumps({"1_data": {"n_repeats": 1, "img_count": images}})
    header = json.dumps({"__metadata__": meta}).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(header)) + header + (payload or path.name.encode() * 64))
    return path


@pytest.fixture
def cfg(tmp_path) -> Config:
    loras_root = tmp_path / "loras"
    (loras_root / "Pony").mkdir(parents=True)
    project = tmp_path / "project"
    project.mkdir()
    (project / "config.toml").write_text(textwrap.dedent(f"""
        [paths]
        loras_root = '{loras_root}'
        source_subdir = "Pony"
        output_dir = "output"
        review_dir = "review"
        blocklist_dir = "blocklist"

        [blocklist]
        scan_dir = "Pony"

        [organize]
        link_mode = "hardlink"

        [deepinfra]
        base_url = "http://invalid.test"
        model = "fake"
        temperature = 0
        max_tokens = 100
        timeout_s = 1
        image_max_side = 64

        [comfy]
        url = "http://127.0.0.1:1"
        checkpoint = "pony.safetensors"
        clip_skip = -2
        steps = 1
        cfg = 1.0
        sampler = "euler"
        scheduler = "normal"
        base_seed = 1
        timeout_s = 5

        [comfy.sizes]
        square = [1024, 1024]
        portrait = [832, 1216]
        landscape = [1216, 832]

        [probe]
        count = 2
        baseline_count = 1
        weight = 0.9
        aspect = "square"
        prefix = "score_9"
        negative = "lowres"
        templates = ["{{triggers}}, {{top_tags}}", "{{triggers}}"]

        [civitai]
        enabled = false

        [examples]
        max_count = 2
        min_count = 1
    """), encoding="utf-8")
    (project / ".env").write_text("DEEPINFRA_API_KEY=test-key\n", encoding="utf-8")
    c = Config(project / "config.toml")
    assert REAL_PROJECT not in c.output_dir.parents  # never the real project
    return c


@pytest.fixture
def library(cfg):
    """A small library covering each safety case."""
    src = cfg.source_dir
    many = {f"tag{i}": 20 for i in range(80)}  # push rare tags out of the top 60
    return {
        "clean": write_lora(src / "styles/anime/clean_style.safetensors",
                            {"clean style": 20, "1girl": 18, "smile": 10}),
        "csam": write_lora(src / "concepts/bad/csam_concept.safetensors",
                           {"thing": 20, "1girl": 20, "nude": 15, "loli": 3}),
        "rare_csam": write_lora(src / "concepts/bad/rare_tag.safetensors", {**many, "loli": 1}),
        "child_sexual": write_lora(src / "concepts/bad/child_sexual.safetensors",
                                   {"thing": 20, "sex": 10, "child": 2}),
        "child_sfw": write_lora(src / "styles/artists/rockwell.safetensors",
                                {"rockwell style": 20, "children": 8, "hat": 5}),
        "aged": write_lora(src / "concepts/bad/aged.safetensors", {"thing": 20, "9yo": 5, "sex": 12}),
    }


# ---------------------------------------------------------------- fakes

def png(color=(120, 90, 200), size=(64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


class FakeComfy(comfy_mod.Comfy):
    """Real workflow building (so hardening is exercised), fake rendering."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.workflows = []
        self.version = "test"
        self.lora_names = set()
        self._cfg = cfg

    def check(self):
        self.lora_names = {str(p.relative_to(self._cfg.loras_root)).replace("/", "\\")
                           for p in self._cfg.source_dir.rglob("*.safetensors")}

    def render(self, workflow):
        self.workflows.append(workflow)
        return png()


class FakeLLM:
    """Scriptable DeepInfra stand-in. `screen_verdict(index)` decides each screening verdict."""

    def __init__(self, *_a, **_k):
        self.total_cost = 0.0
        self.calls = []
        self.screen_verdict = lambda i: {"verdict": "adult", "sexual": "none", "reason": "ok"}
        self.possible_minor = False
        self.screen_prompts = []
        self.second_opinion_model = None          # set to a name to enable the second opinion
        self.second_verdict = lambda i: {"verdict": "adult", "sexual": "none", "reason": "second: adult"}
        self.second_calls = []                    # number of images sent per second-opinion call
        self.second_error = None

    def image_part(self, _png):
        return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AA=="}}

    def chat_json(self, system, content, max_tokens=None, model=None):
        self.calls.append(system[:40])
        if system in (safety.SCREEN_SYSTEM, safety.SCREEN_SYSTEM_REVIEWED):
            n = sum(1 for p in content if p.get("type") == "image_url")
            if model is not None and model == self.second_opinion_model:
                self.second_calls.append(n)
                if self.second_error:
                    raise self.second_error
                return {"images": [{"index": i, **self.second_verdict(i)} for i in range(n)]}
            self.screen_prompts.append(system)
            return {"images": [{"index": i, **self.screen_verdict(i)} for i in range(n)]}
        if "planned example images" in system:  # evaluate
            return {"possible_minor": self.possible_minor, "per_example": [], "summary": "evaluated"}
        return {"possible_minor": self.possible_minor, "lora_type": "style", "category_path": ["styles", "anime", "test"],
                "summary": "a test lora", "description": "desc", "nsfw_level": "sfw",
                "metrics": {}, "example_plan": {"rationale": "r", "images": [
                    {"purpose": "p", "prompt": "1girl, smile", "aspect": "portrait", "lora_weight": 0.8}]}}


@pytest.fixture
def fakes(cfg, monkeypatch):
    from lora_classifier import actions
    llm = FakeLLM()
    comfy = FakeComfy(cfg)
    monkeypatch.setattr(actions, "Comfy", lambda _cfg: comfy)
    monkeypatch.setattr(actions, "LLM", lambda _cfg: llm)
    return comfy, llm

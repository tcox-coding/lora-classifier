import textwrap

import pytest

from lora_classifier import civitai, metadata, safety
from lora_classifier.config import Config


def test_writes_are_confined_to_tool_folders(cfg):
    cfg.assert_writable(cfg.output_dir / "x")
    cfg.assert_writable(cfg.review_dir / "x")
    cfg.assert_writable(cfg.blocklist_dir / "x")
    with pytest.raises(PermissionError):
        cfg.assert_writable(cfg.source_dir / "x.safetensors")
    with pytest.raises(PermissionError):
        cfg.assert_writable(cfg.output_dir.parent.parent / "elsewhere.txt")


def test_deletion_guard(cfg, library):
    cfg.assert_deletable_lora(library["clean"])
    with pytest.raises(PermissionError):
        cfg.assert_deletable_lora(library["clean"].with_suffix(".json"))
    with pytest.raises(PermissionError):
        cfg.assert_deletable_lora(cfg.output_dir / "x.safetensors")


def test_output_inside_source_is_refused(tmp_path):
    root = tmp_path / "loras"
    (root / "Pony").mkdir(parents=True)
    (tmp_path / "c.toml").write_text(textwrap.dedent(f"""
        [paths]
        loras_root = '{root}'
        source_subdir = "Pony"
        output_dir = '{root / "Pony" / "out"}'
        [organize]
        link_mode = "none"
        [deepinfra]
        [comfy]
        [probe]
        [examples]
    """), encoding="utf-8")
    with pytest.raises(ValueError):
        Config(tmp_path / "c.toml")


def test_rare_minor_tag_beyond_top_60_is_detected(cfg, library):
    meta = metadata.extract(library["rare_csam"], cfg)
    assert "loli" not in meta["tag_frequency"]  # not among the top 60...
    assert meta["minor_tag_hits"] == {"loli": 1}  # ...but still caught
    assert safety.assess(meta)["tier"] == safety.CAUTION


def test_extract_basic_fields(cfg, library):
    meta = metadata.extract(library["clean"], cfg)
    assert meta["triggers"] == ["clean style"]
    assert meta["comfy_name"] == "Pony\\styles\\anime\\clean_style.safetensors"
    assert meta["sexual_tag_hits"] == {} and meta["minor_tag_hits"] == {}


def test_triggers_never_include_minor_terms(cfg):
    p = cfg.source_dir / "x/minor_trigger.safetensors"
    from tests.conftest import write_lora
    write_lora(p, {"little girl": 20, "hat": 5})
    assert all(not safety.minor_terms_in(t) for t in metadata.extract(p, cfg)["triggers"])


@pytest.mark.parametrize("w,h,bucket", [(1280, 1920, (832, 1216)), (1920, 1080, (1344, 768)),
                                        (512, 512, (1024, 1024)), (2000, 700, (1536, 640))])
def test_nearest_bucket(w, h, bucket):
    assert civitai.nearest_bucket(w, h) == bucket


def test_bucket_plan_proportions_and_fallback():
    info = {"images": [{"bucket": [832, 1216]}] * 7 + [{"bucket": [1024, 1024]}] * 3}
    plan = civitai.bucket_plan(info, 8)
    assert len(plan) == 8 and plan.count((832, 1216)) == 6 and plan.count((1024, 1024)) == 2
    assert plan[:2] == [(832, 1216), (1024, 1024)]  # every size covered early
    assert civitai.bucket_plan(None, 3) == [(1024, 1024)] * 3

import json
import shutil

import pytest

from lora_classifier import blocklist, hashing, metadata, reviews, safety
from lora_classifier.store import load_state, save_state


def _meta(cfg, path):
    m = metadata.extract(path, cfg)
    m["sha256"] = hashing.sha256_of(path)
    return m


# ---------------------------------------------------------------- reviews

def test_approval_is_bound_to_hash_and_logged(cfg, library):
    m = _meta(cfg, library["child_sfw"])
    a = safety.assess(m)
    reviews.approve(cfg, m["sha256"], m, "non-sexual illustration style", a)
    assert reviews.approval(cfg, m["sha256"])["notes"] == "non-sexual illustration style"
    assert reviews.approval(cfg, "0" * 64) is None
    assert [e["event"] for e in reviews.history(cfg, m["sha256"])] == ["approved"]
    assert reviews.revoke(cfg, m["sha256"], m, "changed my mind")
    assert reviews.approval(cfg, m["sha256"]) is None
    assert reviews.history(cfg, m["sha256"])[-1]["event"] == "approval revoked"


def test_flagged_lora_can_be_approved(cfg, library):
    m = _meta(cfg, library["csam"])  # caution tier: still reviewable
    reviews.approve(cfg, m["sha256"], m, "autotagger noise on adults", safety.assess(m))
    assert reviews.approval(cfg, m["sha256"])


def test_manual_quarantine_blocks_approval_until_released(cfg, library):
    m = _meta(cfg, library["clean"])
    reviews.approve(cfg, m["sha256"], m, "ok", safety.assess(m))
    reviews.quarantine(cfg, m["sha256"], m, "looked wrong")
    assert reviews.quarantine_record(cfg, m["sha256"])["notes"] == "looked wrong"
    assert reviews.approval(cfg, m["sha256"]) is None  # quarantining revokes the approval
    with pytest.raises(PermissionError):
        reviews.approve(cfg, m["sha256"], m, "again", safety.assess(m))
    assert reviews.release(cfg, m["sha256"], m, "mistake")
    reviews.approve(cfg, m["sha256"], m, "again", safety.assess(m))
    events = [e["event"] for e in reviews.history(cfg, m["sha256"])]
    assert events == ["approved", "approval revoked", "quarantined", "released from quarantine", "approved"]


# ---------------------------------------------------------------- blocklist

def test_block_deletes_file_link_and_outputs(cfg, library):
    src = library["clean"]
    m = _meta(cfg, src)
    link = cfg.loras_out / "styles/clean_style.safetensors"
    link.parent.mkdir(parents=True)
    link.hardlink_to(src)
    img_dir = cfg.examples_dir / m["id"]
    img_dir.mkdir(parents=True)
    (img_dir / "probe_01.png").write_bytes(b"x")
    save_state(cfg, {"meta": m, "status": "done", "stages": {}, "safety": {},
                     "placement": {"lora": str(link), "category_path": ["styles"], "counted": False}})

    entry = blocklist.block(cfg, m, "test block")

    assert not src.exists() and not link.exists() and not img_dir.exists()
    assert blocklist.is_blocked(cfg, m["sha256"]) and entry["reason"] == "test block"
    assert load_state(cfg, m["id"])["status"] == "blocked"
    assert "BLOCKED" in blocklist.read_log(cfg)
    assert not link.parent.exists()  # empty category folder cleaned up


def test_block_refuses_changed_file(cfg, library):
    m = _meta(cfg, library["clean"])
    m["sha256"] = "f" * 64  # recorded hash no longer matches the file
    with pytest.raises(PermissionError):
        blocklist.block(cfg, m, "x")
    assert library["clean"].exists() and not blocklist.entries(cfg)


def test_enforce_finds_redownloaded_copy(cfg, library, tmp_path):
    src = library["clean"]
    backup = tmp_path / "copy.bin"
    shutil.copy2(src, backup)
    m = _meta(cfg, src)
    blocklist.block(cfg, m, "bad")
    # The same file downloaded again, under a new name and folder.
    again = cfg.source_dir / "downloads/renamed.safetensors"
    again.parent.mkdir(parents=True)
    shutil.copy2(backup, again)
    other = library["child_sfw"]

    found = blocklist.enforce(cfg, dry_run=True)
    assert [f["path"] for f in found] == [str(again)] and again.exists()  # dry run deletes nothing

    found = blocklist.enforce(cfg)
    assert found[0]["deleted"] and not again.exists()
    assert other.exists()  # different hash untouched
    assert "AUTO-REJECTED" in blocklist.read_log(cfg)
    entry = json.loads((cfg.blocklist_dir / f"{m['sha256']}.json").read_text(encoding="utf-8"))
    assert entry["rejections"][0]["path"] == str(again)


def test_delete_verified_refuses_hash_mismatch(cfg, library):
    with pytest.raises(PermissionError):
        blocklist.delete_verified(cfg, library["clean"], "0" * 64)
    assert library["clean"].exists()


def test_delete_verified_refuses_non_lora_paths(cfg, tmp_path):
    victim = tmp_path / "important.txt"
    victim.write_text("keep me")
    with pytest.raises(PermissionError):
        blocklist.delete_verified(cfg, victim, hashing.sha256_of(victim))
    assert victim.exists()

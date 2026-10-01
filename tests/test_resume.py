"""Crash recovery: a run that dies part-way must continue where it left off, and files damaged by
a crash or power loss must never stop the program or weaken a safety decision."""
import json

import pytest

from lora_classifier import actions, blocklist, hashing, metadata, organize, reviews, store
from lora_classifier.taxonomy import Taxonomy


class SimulatedCrash(BaseException):
    """Like pulling the plug: not an Exception, so nothing in the program catches it."""


def lid(cfg, path):
    return metadata.lora_id(path.relative_to(cfg.source_dir).as_posix())


def rendered_loras(comfy):
    return [wf["2"]["inputs"]["lora_name"] for wf in comfy.workflows if "2" in wf]


def crash_after(comfy, n):
    real = comfy.render

    def render(wf):
        if len(comfy.workflows) >= n:
            raise SimulatedCrash()
        return real(wf)
    comfy.render = render
    return real


def test_crash_mid_run_then_continue(cfg, library, fakes):
    comfy, llm = fakes
    # Every LoRA renders 3 probes + 1 example; the held one renders nothing. Crash on render 10:
    # two LoRAs finish, the third dies with its probes rendered in memory.
    real = crash_after(comfy, 9)
    with pytest.raises(SimulatedCrash):
        actions.run(cfg)
    first = actions.progress(cfg)
    assert first["processed"] >= 2 and first["remaining"] >= 1
    finished = {s["meta"]["source_path"] for s in organize.all_states(cfg) if s["status"] == "done"}

    comfy.render = real
    comfy.workflows.clear()
    counts = actions.run(cfg)  # no "redo": continues where it left off
    assert actions.progress(cfg)["remaining"] == 0
    names = rendered_loras(comfy)
    for path in finished:  # nothing that finished before the crash is rendered again
        assert not any(n.endswith(path.split("Pony\\")[-1].split("Pony/")[-1].replace("/", "\\")) for n in names)
    assert sum(counts.values()) == first["remaining"]


def test_interrupted_lora_restarts_clean(cfg, library, fakes):
    comfy, llm = fakes
    src = library["clean"]
    m = metadata.extract(src, cfg)
    # What a crash after the probe stage leaves behind: an in_progress record and stale images.
    stale = cfg.examples_dir / m["id"]
    stale.mkdir(parents=True)
    (stale / "example_07.png").write_bytes(b"left over")
    store.save_state(cfg, {"meta": m, "status": "in_progress", "stages": {}, "safety": {"screenings": []}})
    assert actions.progress(cfg, "clean_style")["interrupted"] == 1
    actions.run(cfg, match="clean_style")
    s = store.load_state(cfg, m["id"])
    assert s["status"] == "done"
    assert not (stale / "example_07.png").exists()  # stale file from the dead attempt is gone
    assert sorted(p.name for p in stale.iterdir()) == ["baseline_01.png", "example_01.png",
                                                       "probe_01.png", "probe_02.png"]


@pytest.mark.parametrize("damage", [b"", b"{\"meta\": {\"id\": ", b"\x00\x00\x00\x00", b"[]"])
def test_damaged_state_file_is_set_aside_and_redone(cfg, library, fakes, damage):
    comfy, _ = fakes
    actions.run(cfg, match="clean_style")
    p = store.state_path(cfg, lid(cfg, library["clean"]))
    p.write_bytes(damage)  # e.g. zero-length after a power cut
    states = organize.all_states(cfg)  # must not raise
    assert all(s["meta"]["id"] != p.stem for s in states)
    assert list(cfg.state_dir.glob(p.name + ".corrupt-*"))  # kept for inspection
    assert actions.progress(cfg, "clean_style")["not_started"] == 1
    comfy.workflows.clear()
    actions.run(cfg, match="clean_style")
    assert store.load_state(cfg, p.stem)["status"] == "done" and comfy.workflows


def test_damaged_taxonomy_is_rebuilt_from_records(cfg, library, fakes):
    actions.run(cfg, match="clean_style")
    before = Taxonomy(cfg).paths
    cfg.taxonomy_file.write_text('{"styles/anime": 3', encoding="utf-8")  # truncated
    rebuilt = Taxonomy(cfg).paths
    assert rebuilt == before
    actions.run(cfg, match="rockwell")  # and a run still works


def test_damaged_quarantine_file_is_rebuilt_from_the_log(cfg, library, fakes):
    """A quarantine must survive its file being damaged: otherwise the LoRA would be rendered."""
    comfy, _ = fakes
    actions.run(cfg, match="csam_concept")
    q = lid(cfg, library["csam"])
    actions.quarantine(cfg, q, "no")
    (cfg.review_dir / "quarantine.json").write_text("{\"abc", encoding="utf-8")
    sha = hashing.sha256_of(library["csam"])
    assert reviews.quarantine_record(cfg, sha)["notes"] == "no"
    comfy.workflows.clear()
    actions.reprocess(cfg, q)
    assert not comfy.workflows and store.load_state(cfg, q)["status"] == "quarantined"


def test_damaged_approvals_file_is_rebuilt_from_the_log(cfg, library, fakes):
    actions.approve(cfg, lid(cfg, library["child_sfw"]), "fine")
    (cfg.review_dir / "approvals.json").write_bytes(b"")
    assert reviews.approval(cfg, hashing.sha256_of(library["child_sfw"]))["notes"] == "fine"


def test_review_log_line_cut_short_is_skipped(cfg, library, fakes):
    actions.approve(cfg, lid(cfg, library["child_sfw"]), "fine")
    with open(cfg.review_dir / "review_log.jsonl", "a", encoding="utf-8") as fh:
        fh.write('{"time": "2026-09-25", "sha256": "x", "ev')  # crash mid-append
    sha = hashing.sha256_of(library["child_sfw"])
    assert [e["event"] for e in reviews.history(cfg, sha)] == ["approved"]
    actions.revoke(cfg, lid(cfg, library["child_sfw"]))  # appending after the damage still works
    assert [e["event"] for e in reviews.history(cfg, sha)] == ["approved", "approval revoked"]


def test_damaged_blocklist_entry_still_blocks(cfg, library, fakes, tmp_path):
    import shutil
    src = library["clean"]
    keep = tmp_path / "copy.bin"
    shutil.copy2(src, keep)
    blocklist.block(cfg, metadata.extract(src, cfg), "bad")
    sha = hashing.sha256_of(keep)
    (cfg.blocklist_dir / f"{sha}.json").write_text("{\"sha", encoding="utf-8")
    shutil.copy2(keep, src)  # downloaded again
    assert blocklist.is_blocked(cfg, sha)
    actions.run(cfg)
    assert not src.exists()


def test_writes_leave_no_temp_files(cfg, library, fakes):
    actions.run(cfg)
    actions.approve(cfg, lid(cfg, library["child_sfw"]), "ok")
    for root in (cfg.output_dir, cfg.review_dir):
        assert not list(root.rglob("*.tmp"))


def test_progress_counts_and_run_log(cfg, library, fakes, quiet_log):
    actions.run(cfg, limit=2)
    p = actions.progress(cfg)
    assert p["total"] == 6 and p["processed"] == 2 and p["remaining"] == 4
    quiet_log.clear()
    actions.run(cfg)
    assert any("2 of 6 LoRA(s) already processed and skipped; continuing with 4 (4 not started, 0 queued"
               in line for line in quiet_log)
    assert actions.progress(cfg)["remaining"] == 0

"""End-to-end pipeline tests with a fake ComfyUI and fake DeepInfra."""
import shutil

import pytest

from lora_classifier import actions, blocklist, hashing, metadata, organize, reviews, safety
from lora_classifier.pipeline import load_state
from lora_classifier.store import save_state

CHILDLIKE_NONSEXUAL = {"verdict": "possible_minor", "sexual": "none", "reason": "child flying a kite"}
CHILDLIKE_SUGGESTIVE = {"verdict": "possible_minor", "sexual": "suggestive", "reason": "bad"}
ADULT = {"verdict": "adult", "sexual": "none", "reason": "ok"}


def lora_id(cfg, path):
    return metadata.lora_id(path.relative_to(cfg.source_dir).as_posix())


def state_of(cfg, path):
    return load_state(cfg, lora_id(cfg, path))


def approve(cfg, path, notes="reviewed"):
    actions.approve(cfg, lora_id(cfg, path), notes)


def negatives(comfy):
    return [wf["5"]["inputs"]["text"] for wf in comfy.workflows]


def positives(comfy):
    return [wf["4"]["inputs"]["text"] for wf in comfy.workflows]


# ---------------------------------------------------------------- automatic flags

def test_clean_lora_is_classified_in_standard_mode(cfg, library, fakes):
    comfy, llm = fakes
    actions.run(cfg, match="clean_style")
    s = state_of(cfg, library["clean"])
    assert s["status"] == "done"
    assert s["render_settings"]["safety_mode"] == safety.STANDARD
    assert len(s["stages"]["probe"]["images"]) == 3  # 2 probes + 1 baseline
    assert all(p.startswith("adult") for p in positives(comfy))
    assert all("loli" in n and "(loli:1.6)" not in n for n in negatives(comfy))
    assert "styles" in s["placement"]["lora"] and library["clean"].exists()


@pytest.mark.parametrize("key", ["csam", "rare_csam", "child_sexual"])
def test_caution_lora_is_processed_in_strict_mode(cfg, library, fakes, key):
    comfy, _ = fakes
    actions.run(cfg, match=library[key].stem)
    s = state_of(cfg, library[key])
    assert s["status"] == "done", s["safety"].get("high_risk_summary")
    assert s["safety"]["assessment"]["tier"] == safety.CAUTION
    assert s["render_settings"]["safety_mode"] == safety.STRICT
    assert comfy.workflows and all("(loli:1.6)" in n and "(child:1.5)" in n for n in negatives(comfy))
    assert "caution loras" not in s["placement"]["lora"]  # placed in its real category
    item = next(i for i in actions.review_items(cfg) if i["id"] == s["meta"]["id"])
    assert item["tier"] == safety.CAUTION and "Caution:" in item["summary"]


def test_reviewable_lora_is_processed_in_strict_mode(cfg, library, fakes):
    comfy, _ = fakes
    actions.run(cfg, match="rockwell")
    s = state_of(cfg, library["child_sfw"])
    assert s["status"] == "done" and s["safety"]["assessment"]["tier"] == safety.REVIEWABLE
    assert s["render_settings"]["safety_mode"] == safety.STRICT and comfy.workflows


def test_held_lora_is_not_rendered_until_approved(cfg, library, fakes):
    comfy, llm = fakes
    actions.run(cfg, match="aged")
    s = state_of(cfg, library["aged"])
    assert s["status"] == "caution" and not comfy.workflows and "Held" in s["safety"]["high_risk_summary"]
    assert "caution loras" in s["placement"]["description_md"]
    approve(cfg, library["aged"], "checked: tagging error on adults")
    assert state_of(cfg, library["aged"])["status"] == "pending"  # approval sends it back to Run
    actions.run(cfg, match="aged")  # a normal Continue picks it up
    s = state_of(cfg, library["aged"])
    assert s["status"] == "done" and s["render_settings"]["safety_mode"] == safety.REVIEWED_ADULT
    assert all(p.startswith("(adult:1.3), (mature:1.2)") for p in positives(comfy))


def test_screener_flag_sets_caution_and_saves_nothing(cfg, library, fakes):
    comfy, llm = fakes
    llm.screen_verdict = lambda i: CHILDLIKE_NONSEXUAL if i == 0 else ADULT
    actions.run(cfg, match="csam_concept")
    s = state_of(cfg, library["csam"])
    assert s["status"] == "caution"
    assert not (cfg.examples_dir / s["meta"]["id"]).exists()
    prompts = s["safety"]["high_risk_prompts"]
    assert [p["flagged"] for p in prompts] == [True, False, False]
    assert all(p["seed"] is not None and p["width"] and p["mode"] == safety.STRICT for p in prompts)
    assert "caution loras" in s["placement"]["description_md"]


# ---------------------------------------------------------------- manual review

def test_approved_sfw_lora_keeps_childlike_nonsexual_images(cfg, library, fakes):
    comfy, llm = fakes
    approve(cfg, library["child_sfw"])
    llm.screen_verdict = lambda i: CHILDLIKE_NONSEXUAL
    llm.possible_minor = True  # classifier notices children: allowed in reviewed SFW mode
    actions.run(cfg, match="rockwell")
    s = state_of(cfg, library["child_sfw"])
    assert s["status"] == "done", s["safety"].get("high_risk_summary")
    assert s["render_settings"]["safety_mode"] == safety.REVIEWED_SFW
    assert s["review"]["notes"] == "reviewed"
    assert all(p.startswith("rating_safe") for p in positives(comfy))
    assert all("(nsfw:1.5)" in n for n in negatives(comfy))


def test_approval_is_never_revoked_automatically(cfg, library, fakes):
    """Flagged images of an approved LoRA are discarded one by one; the rest are kept, the
    approval stays and the LoRA is classified normally."""
    comfy, llm = fakes
    approve(cfg, library["child_sfw"])
    llm.screen_verdict = lambda i: CHILDLIKE_SUGGESTIVE if i == 0 else ADULT
    actions.run(cfg, match="rockwell")
    s = state_of(cfg, library["child_sfw"])
    sha = hashing.sha256_of(library["child_sfw"])
    assert s["status"] == "done"
    assert reviews.approval(cfg, sha) is not None
    assert not any("revoked" in e["event"] for e in reviews.history(cfg, sha))
    # index 0 of each batch is flagged: probe 1 and the single example are discarded (never saved);
    # probe 2 and the baseline are kept
    img_dir = cfg.examples_dir / s["meta"]["id"]
    assert sorted(p.name for p in img_dir.iterdir()) == ["baseline_01.png", "probe_01.png"]
    assert len(s["stages"]["probe"]["images"]) == 2
    discarded = s["safety"]["discarded_prompts"]
    assert len(discarded) == 2 and all(d["flagged"] for d in discarded)  # probe #1 and example #1 (index 0)
    assert "high_risk_prompts" not in s["safety"]
    assert "Discarded by the screener (2)" in open(s["placement"]["description_md"], encoding="utf-8").read()


def test_approved_lora_with_every_probe_discarded_is_still_classified(cfg, library, fakes):
    comfy, llm = fakes
    approve(cfg, library["child_sexual"])
    llm.screen_verdict = lambda i: CHILDLIKE_SUGGESTIVE  # childlike AND sexual: always discarded
    actions.run(cfg, match="child_sexual")
    s = state_of(cfg, library["child_sexual"])
    assert s["status"] == "needs_review" and s["stages"]["analysis"]["analysis_mode"] == "metadata_only"
    assert "caution loras" not in s["placement"]["lora"]  # stays in the normal list
    assert reviews.approval(cfg, hashing.sha256_of(library["child_sexual"])) is not None
    assert not (cfg.examples_dir / s["meta"]["id"]).exists() or not any((cfg.examples_dir / s["meta"]["id"]).iterdir())
    assert all(p.startswith("(adult:1.3), (mature:1.2)") for p in positives(comfy))


def test_approved_explicit_lora_keeps_childlike_nonsexual_images(cfg, library, fakes):
    comfy, llm = fakes
    approve(cfg, library["child_sexual"])
    llm.screen_verdict = lambda i: CHILDLIKE_NONSEXUAL
    actions.run(cfg, match="child_sexual")
    s = state_of(cfg, library["child_sexual"])
    assert s["status"] == "done" and s["render_settings"]["safety_mode"] == safety.REVIEWED_ADULT
    assert "discarded_prompts" not in s["safety"] and len(s["stages"]["probe"]["images"]) == 3
    assert set(llm.screen_prompts) == {safety.SCREEN_SYSTEM_REVIEWED}


def test_unapproved_lora_is_screened_with_the_strict_prompt(cfg, library, fakes):
    comfy, llm = fakes
    actions.run(cfg, match="csam_concept")
    assert llm.screen_prompts and set(llm.screen_prompts) == {safety.SCREEN_SYSTEM}


def test_screening_error_for_approved_lora_discards_that_chunk_only(cfg, library, fakes, monkeypatch):
    from lora_classifier import pipeline
    comfy, llm = fakes
    monkeypatch.setattr(pipeline, "SCREEN_CHUNK", 1)  # screen each image separately
    approve(cfg, library["child_sfw"])
    real = safety.screen
    calls = {"n": 0}

    def flaky(llm_, images, mode):
        calls["n"] += 1
        if calls["n"] == 1:
            raise safety.ScreeningError("refused")
        return real(llm_, images, mode)
    monkeypatch.setattr(safety, "screen", flaky)
    actions.run(cfg, match="rockwell")
    s = state_of(cfg, library["child_sfw"])
    assert s["status"] == "done" and len(s["stages"]["probe"]["images"]) == 2
    assert s["safety"]["discarded_prompts"][0]["verdict"] == "unverified"


def test_unapproved_lora_is_still_set_aside_on_any_flag(cfg, library, fakes):
    comfy, llm = fakes
    llm.screen_verdict = lambda i: CHILDLIKE_SUGGESTIVE if i == 1 else ADULT
    actions.run(cfg, match="rockwell")
    s = state_of(cfg, library["child_sfw"])
    assert s["status"] == "caution" and not (cfg.examples_dir / s["meta"]["id"]).exists()


def test_approving_a_set_aside_lora_queues_it(cfg, library, fakes):
    comfy, llm = fakes
    llm.screen_verdict = lambda i: CHILDLIKE_NONSEXUAL
    actions.run(cfg, match="clean_style")
    assert state_of(cfg, library["clean"])["status"] == "caution"
    approve(cfg, library["clean"], "false positive")
    s = state_of(cfg, library["clean"])
    assert s["status"] == "pending" and s["placement"] is None
    assert actions.progress(cfg, "clean_style")["queued"] == 1


def test_manual_quarantine_and_release(cfg, library, fakes):
    comfy, llm = fakes
    actions.run(cfg, match="csam_concept")
    lid = lora_id(cfg, library["csam"])
    assert state_of(cfg, library["csam"])["status"] == "done"
    actions.quarantine(cfg, lid, "not comfortable with this one")
    s = state_of(cfg, library["csam"])
    assert s["status"] == "quarantined" and not (cfg.examples_dir / lid).exists()
    assert "quarantined loras" in s["placement"]["lora"] and library["csam"].exists()
    rendered = len(comfy.workflows)
    actions.reprocess(cfg, lid)  # a quarantined LoRA is never rendered, even when asked
    assert len(comfy.workflows) == rendered and state_of(cfg, library["csam"])["status"] == "quarantined"
    with pytest.raises(PermissionError):
        actions.approve(cfg, lid, "please")
    with pytest.raises(PermissionError):
        actions.test(cfg, lid, positive="1girl")
    actions.release(cfg, lid, "second look: fine")
    assert state_of(cfg, library["csam"])["status"] == "pending"
    actions.run(cfg, match="csam_concept")  # pending LoRAs are picked up by a normal run
    assert state_of(cfg, library["csam"])["status"] == "done"


def test_blocked_hash_is_deleted_before_a_run(cfg, library, fakes, tmp_path):
    comfy, _ = fakes
    src = library["clean"]
    keep = tmp_path / "copy.bin"
    shutil.copy2(src, keep)
    blocklist.block(cfg, metadata.extract(src, cfg), "never again")
    shutil.copy2(keep, src)  # "downloaded again"
    actions.run(cfg)
    assert not src.exists()
    assert "AUTO-REJECTED" in blocklist.read_log(cfg)
    assert not any("clean_style" in wf["2"]["inputs"]["lora_name"] for wf in comfy.workflows if "2" in wf)


def test_stop_leaves_lora_unfinished(cfg, library, fakes):
    comfy, _ = fakes
    counts = actions.run(cfg, match="clean_style", should_stop=lambda: True)
    assert counts.get("stopped") and state_of(cfg, library["clean"]) is None and not comfy.workflows


# ---------------------------------------------------------------- test bench

def test_test_render_allowed_for_flagged_lora_in_strict_mode(cfg, library, fakes):
    comfy, _ = fakes
    actions.run(cfg, match="csam_concept")
    out = actions.test(cfg, lora_id(cfg, library["csam"]), positive="1girl, loli, smile", count=1)
    assert out["passed"] and out["mode"] == safety.STRICT
    assert "loli" not in comfy.workflows[-1]["4"]["inputs"]["text"]


def test_test_render_requires_approval_when_held_or_set_aside(cfg, library, fakes):
    comfy, llm = fakes
    actions.run(cfg, match="aged")
    with pytest.raises(PermissionError):
        actions.test(cfg, lora_id(cfg, library["aged"]), positive="1girl")
    llm.screen_verdict = lambda i: CHILDLIKE_NONSEXUAL
    actions.run(cfg, match="clean_style")  # screener flags it -> set aside
    with pytest.raises(PermissionError):
        actions.test(cfg, lora_id(cfg, library["clean"]), positive="1girl")


def test_test_render_passes_and_saves(cfg, library, fakes):
    comfy, llm = fakes
    approve(cfg, library["child_sfw"])
    llm.screen_verdict = lambda i: CHILDLIKE_NONSEXUAL
    out = actions.test(cfg, lora_id(cfg, library["child_sfw"]), positive="child, flying a kite, nude",
                       count=2, seed=5, aspect="832x1216", save=True)
    assert out["passed"] and out["mode"] == safety.REVIEWED_SFW
    assert all(r["png"] and r["path"] for r in out["results"])
    assert "nude" not in comfy.workflows[-1]["4"]["inputs"]["text"]  # sexual tag stripped
    assert comfy.workflows[-1]["6"]["inputs"]["width"] == 832
    events = [e["event"] for e in reviews.history(cfg, hashing.sha256_of(library["child_sfw"]))]
    assert "test render" in events


def test_test_render_for_approved_lora_hides_flagged_and_keeps_approval(cfg, library, fakes):
    comfy, llm = fakes
    actions.run(cfg, match="rockwell")
    approve(cfg, library["child_sfw"])
    llm.screen_verdict = lambda i: CHILDLIKE_SUGGESTIVE if i == 1 else CHILDLIKE_NONSEXUAL
    out = actions.test(cfg, lora_id(cfg, library["child_sfw"]), positive="child", count=2, save=True)
    assert not out["passed"] and out["approved"]
    flagged, ok = out["results"][1], out["results"][0]
    assert flagged["png"] is None and flagged["path"] is None  # never shown or saved
    assert ok["png"] and ok["path"]
    assert len(list((cfg.tests_dir / lora_id(cfg, library["child_sfw"])).iterdir())) == 1
    assert reviews.approval(cfg, hashing.sha256_of(library["child_sfw"])) is not None
    assert state_of(cfg, library["child_sfw"])["status"] == "done"
    assert "high_risk_summary" not in out


def test_test_render_for_unapproved_lora_discards_all_and_sets_aside(cfg, library, fakes):
    comfy, llm = fakes
    actions.run(cfg, match="csam_concept")
    llm.screen_verdict = lambda i: CHILDLIKE_NONSEXUAL if i == 1 else ADULT
    out = actions.test(cfg, lora_id(cfg, library["csam"]), positive="1girl", count=2, save=True)
    assert not out["passed"] and all(r["png"] is None for r in out["results"])
    assert state_of(cfg, library["csam"])["status"] == "caution"


# ---------------------------------------------------------------- older records

def test_legacy_high_risk_record_becomes_caution(cfg, library, fakes):
    m = metadata.extract(library["csam"], cfg)
    save_state(cfg, {"meta": m, "status": "high_risk", "stages": {},
                     "safety": {"high_risk_summary": "old", "high_risk_detail": "metadata gate: old rules"}})
    assert load_state(cfg, m["id"])["status"] == "caution"
    item = next(i for i in actions.review_items(cfg) if i["id"] == m["id"])
    assert item["tier"] == safety.CAUTION and "re-tier" in item["summary"]
    actions.approve(cfg, m["id"], "fine")  # nothing is locked automatically any more


def test_retier_queues_metadata_only_set_asides(cfg, library, fakes):
    comfy, llm = fakes
    m = metadata.extract(library["csam"], cfg)
    save_state(cfg, {"meta": m, "status": "high_risk", "stages": {}, "placement": None,
                     "safety": {"high_risk_summary": "old", "high_risk_detail": "metadata gate: old rules"}})
    screened = metadata.extract(library["clean"], cfg)
    save_state(cfg, {"meta": screened, "status": "high_risk", "stages": {}, "placement": None,
                     "safety": {"high_risk_summary": "x", "high_risk_detail": "screening flagged (standard): ..."}})
    held = metadata.extract(library["aged"], cfg)
    save_state(cfg, {"meta": held, "status": "high_risk", "stages": {}, "placement": None,
                     "safety": {"high_risk_summary": "x", "high_risk_detail": "metadata gate: old"}})

    changes = actions.retier(cfg, dry_run=True)
    assert [(c["id"], c["to"]) for c in changes] == [(m["id"], "pending")]
    assert load_state(cfg, m["id"])["status"] == "caution"  # dry run changed nothing
    actions.retier(cfg, dry_run=False)
    assert load_state(cfg, m["id"])["status"] == "pending"
    assert load_state(cfg, screened["id"])["status"] == "caution"  # screener flags stay set aside
    assert load_state(cfg, held["id"])["status"] == "caution"      # held stays held
    actions.run(cfg, match="csam_concept")
    assert load_state(cfg, m["id"])["status"] == "done" and comfy.workflows


def test_done_lora_that_is_now_held_moves_to_caution_on_retier(cfg, library, fakes):
    held = metadata.extract(library["aged"], cfg)
    save_state(cfg, {"meta": held, "status": "done", "stages": {"analysis": {"category_path": ["x"]}},
                     "safety": {"screenings": []}, "placement": None})
    assert [c["to"] for c in actions.retier(cfg)] == ["caution"]
    actions.retier(cfg, dry_run=False)
    assert load_state(cfg, held["id"])["status"] == "caution"


# ---------------------------------------------------------------- maintenance

def test_rebuild_is_idempotent(cfg, library, fakes):
    actions.run(cfg)
    listing = sorted(p.relative_to(cfg.output_dir).as_posix() for p in (cfg.output_dir / "loras").rglob("*"))
    actions.rebuild(cfg)
    again = sorted(p.relative_to(cfg.output_dir).as_posix() for p in (cfg.output_dir / "loras").rglob("*"))
    assert listing == again
    for p in library.values():
        assert p.exists()  # rebuilding never touches the source


def test_status_counts(cfg, library, fakes):
    actions.run(cfg)
    counts = actions.status(cfg)
    assert counts == {"done": 5, "caution": 1}  # only the held LoRA is set aside
    assert len(organize.all_states(cfg)) == 6

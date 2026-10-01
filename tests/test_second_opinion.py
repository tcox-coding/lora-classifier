"""Second opinion: an image the first screener flags is discarded only if an independent model
agrees. Failures of the second model never let an image through."""
import pytest

from lora_classifier import actions, hashing, metadata, reviews, safety
from lora_classifier.llm import LLMError
from lora_classifier.pipeline import load_state
from tests.conftest import FakeLLM, png

SECOND = "Qwen/Qwen3-VL-235B-A22B-Instruct"
CHILD_SUGGESTIVE = {"verdict": "possible_minor", "sexual": "suggestive", "reason": "first: childlike"}
CHILD_NONE = {"verdict": "possible_minor", "sexual": "none", "reason": "first: childlike, clothed"}
ADULT = {"verdict": "adult", "sexual": "explicit", "reason": "adult"}


def llm_with(first, second=None, error=None):
    llm = FakeLLM()
    llm.second_opinion_model = SECOND
    llm.screen_verdict = first
    if second:
        llm.second_verdict = second
    llm.second_error = error
    return llm


# ---------------------------------------------------------------- the rule

def test_only_flagged_images_are_sent_for_a_second_opinion():
    llm = llm_with(lambda i: CHILD_SUGGESTIVE if i in (1, 3) else ADULT)
    safety.screen(llm, [png()] * 5, safety.STRICT)
    assert llm.second_calls == [2]


def test_no_second_call_when_nothing_is_flagged():
    llm = llm_with(lambda i: ADULT)
    r = safety.screen(llm, [png()] * 3, safety.STRICT)
    assert r["passed"] and llm.second_calls == []


def test_second_opinion_overrules_a_hallucinated_flag():
    llm = llm_with(lambda i: CHILD_SUGGESTIVE, lambda i: {"verdict": "adult", "sexual": "suggestive", "reason": "adult woman"})
    r = safety.screen(llm, [png()], safety.STRICT)
    v = r["verdicts"][0]
    assert r["passed"] and v["verdict"] == "adult" and v["first_verdict"] == "possible_minor"
    assert v["second_opinion"]["model"] == SECOND and "overruled" in v["reason"]


def test_both_agree_so_the_image_is_discarded():
    llm = llm_with(lambda i: CHILD_SUGGESTIVE, lambda i: {"verdict": "likely_minor", "sexual": "suggestive", "reason": "child"})
    r = safety.screen(llm, [png()], safety.REVIEWED_ADULT)
    assert not r["passed"] and r["csam_hits"] and r["verdicts"][0]["second_opinion"]["verdict"] == "likely_minor"


def test_more_severe_sexual_rating_wins():
    # First: childlike + explicit. Second: childlike but rates it "none". Combined stays explicit.
    llm = llm_with(lambda i: {"verdict": "possible_minor", "sexual": "explicit", "reason": "x"},
                   lambda i: {"verdict": "possible_minor", "sexual": "none", "reason": "y"})
    r = safety.screen(llm, [png()], safety.REVIEWED_ADULT)
    assert not r["passed"] and r["verdicts"][0]["sexual"] == "explicit"


def test_second_model_rating_more_severe_also_counts():
    # First: childlike, clothed (kept for approved). Second: childlike AND suggestive -> discarded.
    llm = llm_with(lambda i: CHILD_NONE, lambda i: {"verdict": "possible_minor", "sexual": "suggestive", "reason": "y"})
    r = safety.screen(llm, [png()], safety.REVIEWED_ADULT)
    # first verdict alone passes in reviewed mode, so no second opinion is asked
    assert r["passed"] and llm.second_calls == []
    r = safety.screen(llm, [png()], safety.STRICT)  # unapproved: childlike alone is flagged
    assert not r["passed"] and r["verdicts"][0]["sexual"] == "suggestive"


@pytest.mark.parametrize("error", [LLMError("HTTP 503"), safety.RefusalError("no"), ValueError("weird")])
def test_second_model_failure_keeps_the_first_flag(error):
    llm = llm_with(lambda i: CHILD_SUGGESTIVE, error=error)
    r = safety.screen(llm, [png()], safety.STRICT)
    assert not r["passed"] and "error" in r["verdicts"][0]["second_opinion"]


def test_incomplete_second_answer_keeps_the_first_flag():
    llm = llm_with(lambda i: CHILD_SUGGESTIVE)
    real = llm.chat_json

    def partial(system, content, max_tokens=None, model=None):
        if model == SECOND:
            return {"images": []}
        return real(system, content, max_tokens, model)
    llm.chat_json = partial
    r = safety.screen(llm, [png(), png()], safety.STRICT)
    assert not r["passed"] and len(r["flagged"]) == 2


def test_unknown_second_verdict_counts_as_agreeing():
    llm = llm_with(lambda i: CHILD_SUGGESTIVE, lambda i: {"verdict": "unsure", "sexual": "suggestive"})
    assert not safety.screen(llm, [png()], safety.STRICT)["passed"]


def test_disabled_when_no_model_configured():
    llm = llm_with(lambda i: CHILD_SUGGESTIVE)
    llm.second_opinion_model = None
    assert not safety.screen(llm, [png()], safety.STRICT)["passed"] and llm.second_calls == []


def test_second_opinion_uses_the_same_mode_prompt():
    llm = llm_with(lambda i: CHILD_SUGGESTIVE)
    seen = []
    real = llm.chat_json

    def spy(system, content, max_tokens=None, model=None):
        if model == SECOND:
            seen.append(system)
        return real(system, content, max_tokens, model)
    llm.chat_json = spy
    safety.screen(llm, [png()], safety.REVIEWED_SFW)
    safety.screen(llm, [png()], safety.STRICT)
    assert seen == [safety.SCREEN_SYSTEM_REVIEWED, safety.SCREEN_SYSTEM]


# ---------------------------------------------------------------- end to end

def lid(cfg, path):
    return metadata.lora_id(path.relative_to(cfg.source_dir).as_posix())


def test_unapproved_lora_is_not_set_aside_when_the_second_opinion_disagrees(cfg, library, fakes):
    comfy, llm = fakes
    llm.second_opinion_model = SECOND
    llm.screen_verdict = lambda i: CHILD_SUGGESTIVE if i == 0 else ADULT
    llm.second_verdict = lambda i: {"verdict": "adult", "sexual": "suggestive", "reason": "petite adult"}
    actions.run(cfg, match="csam_concept")
    s = load_state(cfg, lid(cfg, library["csam"]))
    assert s["status"] == "done" and len(s["stages"]["probe"]["images"]) == 3
    overruled = [v for sc in s["safety"]["screenings"] for v in sc["verdicts"] if v.get("first_verdict")]
    assert overruled and overruled[0]["second_opinion"]["model"] == SECOND


def test_unapproved_lora_is_set_aside_when_both_agree(cfg, library, fakes):
    comfy, llm = fakes
    llm.second_opinion_model = SECOND
    llm.screen_verdict = lambda i: CHILD_SUGGESTIVE if i == 0 else ADULT
    llm.second_verdict = lambda i: CHILD_SUGGESTIVE
    actions.run(cfg, match="csam_concept")
    s = load_state(cfg, lid(cfg, library["csam"]))
    assert s["status"] == "caution"
    flagged = [p for p in s["safety"]["high_risk_prompts"] if p["flagged"]]
    assert len(flagged) == 1


def test_approved_lora_keeps_images_the_second_opinion_clears(cfg, library, fakes):
    comfy, llm = fakes
    llm.second_opinion_model = SECOND
    actions.approve(cfg, lid(cfg, library["child_sexual"]), "reviewed")
    llm.screen_verdict = lambda i: CHILD_SUGGESTIVE
    # second model clears the first image of each batch, agrees on the rest
    llm.second_verdict = lambda i: {"verdict": "adult", "sexual": "explicit", "reason": "adult"} if i == 0 else CHILD_SUGGESTIVE
    actions.run(cfg, match="child_sexual")
    s = load_state(cfg, lid(cfg, library["child_sexual"]))
    assert s["status"] == "done" and len(s["stages"]["probe"]["images"]) == 1
    assert reviews.approval(cfg, hashing.sha256_of(library["child_sexual"])) is not None
    assert all(d["flagged"] for d in s["safety"]["discarded_prompts"])


def test_second_opinion_runs_in_the_pipeline(cfg, library, fakes):
    comfy, llm = fakes
    llm.second_opinion_model = SECOND
    llm.screen_verdict = lambda i: CHILD_SUGGESTIVE if i == 0 else ADULT
    actions.run(cfg, match="csam_concept")
    assert llm.second_calls  # the second model was consulted during a real pipeline run

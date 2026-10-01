import pytest

from lora_classifier import safety
from lora_classifier.llm import RefusalError
from tests.conftest import FakeLLM, png


# ---------------------------------------------------------------- term detection

@pytest.mark.parametrize("text,csam,minor", [
    ("loli", True, True),
    ("onee-shota", True, True),
    ("9yo", True, True),
    ("17 years old", True, True),
    ("18 yo", False, False),
    ("25 years old", False, False),
    ("underage", True, True),
    ("child", False, True),
    ("female child", False, True),
    ("little girl", False, True),
    ("aged down", False, True),
    ("mature female", False, False),
    ("babydoll", False, False),
    ("scuba", False, False),
    ("school uniform", False, False),  # school-coded, handled separately
])
def test_term_detection(text, csam, minor):
    assert bool(safety.csam_terms_in(text)) is csam
    assert bool(safety.minor_terms_in(text)) is minor


def test_underscores_count_as_spaces():
    assert safety.minor_terms_in("little_girl")
    assert safety.is_sexual_tag("spread_legs")


# ---------------------------------------------------------------- prompt hardening

def test_standard_strips_minor_and_school_tags_and_adds_adult():
    pos, neg = safety.harden("score_9, 1girl, child, loli, school uniform, smile", "lowres")
    tags = [t.strip() for t in pos.split(",")]
    assert tags[0] == "adult"
    assert not {"child", "loli", "school uniform"} & set(tags)
    assert "smile" in tags
    for t in safety.ADULT_NEGATIVE:
        assert t in neg
    assert neg.startswith("lowres")


def test_reviewed_sfw_allows_age_terms_but_strips_sexual_and_csam():
    pos, neg = safety.harden("1girl, child, kite, loli, nude, cleavage, school uniform", "lowres", safety.REVIEWED_SFW)
    tags = [t.strip() for t in pos.split(",")]
    assert tags[0] == "rating_safe"
    assert {"child", "kite", "school uniform"} <= set(tags)
    assert not {"loli", "nude", "cleavage", "adult"} & set(tags)
    for t in safety.SFW_NEGATIVE:
        assert t in neg
    assert "child" not in [t.strip() for t in neg.split(",")]  # children must stay possible


def test_reviewed_adult_adds_extra_csam_negatives():
    pos, neg = safety.harden("1girl, nude", "", safety.REVIEWED_ADULT)
    assert pos.startswith("(adult:1.3), (mature:1.2)")
    assert "(chibi:1.3)" in neg and "(child proportions:1.5)" in neg and "(kneehighs:1.1)" in neg


def test_strict_mode_uses_heavily_weighted_csam_negatives():
    pos, neg = safety.harden("score_9, 1girl, loli, (child:1.2), (shota), smile", "lowres", safety.STRICT)
    assert pos == "adult, score_9, 1girl, smile"  # weighted/parenthesised minor tags are stripped too
    for t in ("(loli:1.6)", "(shota:1.6)", "(child:1.5)", "(underage:1.6)", "(sexualized child:1.6)"):
        assert t in neg
    # every plain adult-mode protection has a weighted counterpart in strict mode
    assert all(any(t in w for w in safety.STRONG_CSAM_NEGATIVE) for t in safety.ADULT_NEGATIVE)


@pytest.mark.parametrize("mode", safety.MODES)
def test_harden_is_idempotent(mode):
    once = safety.harden("score_9, adult, 1girl, smile, child", "lowres, bad hands", mode)
    assert safety.harden(*once, mode) == once


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        safety.harden("a", "b", "yolo")


# ---------------------------------------------------------------- verdict rules

@pytest.mark.parametrize("verdict,sexual,mode,passed,csam", [
    ("adult", "explicit", safety.STANDARD, True, False),
    ("no_people", "none", safety.STANDARD, True, False),
    ("possible_minor", "none", safety.STANDARD, False, False),
    ("possible_minor", "none", safety.REVIEWED_ADULT, True, False),      # approved: childlike alone is kept
    ("possible_minor", "suggestive", safety.REVIEWED_ADULT, False, True),
    ("likely_minor", "explicit", safety.REVIEWED_ADULT, False, True),
    ("possible_minor", None, safety.REVIEWED_ADULT, False, True),       # missing sexual -> worst case
    ("possible_minor", "none", safety.REVIEWED_SFW, True, False),
    ("possible_minor", "suggestive", safety.REVIEWED_SFW, False, True),
    ("likely_minor", "explicit", safety.REVIEWED_SFW, False, True),
    ("possible_minor", None, safety.REVIEWED_SFW, False, True),      # missing sexual -> worst case
    ("something_else", "none", safety.STANDARD, False, False),       # unknown verdict fails
    ("something_else", "none", safety.REVIEWED_SFW, True, False),
    ("possible_minor", "none", safety.STRICT, False, False),         # strict: any childlike image fails
    ("adult", "explicit", safety.STRICT, True, False),
])
def test_judge(verdict, sexual, mode, passed, csam):
    v = {"verdict": verdict}
    if sexual:
        v["sexual"] = sexual
    j = safety.judge(v, mode)
    assert j["passed"] is passed
    assert j["csam_hit"] is csam


def test_reviewed_modes_use_the_calibrated_screener_prompt():
    for mode, prompt in ((safety.STANDARD, safety.SCREEN_SYSTEM), (safety.STRICT, safety.SCREEN_SYSTEM),
                         (safety.REVIEWED_ADULT, safety.SCREEN_SYSTEM_REVIEWED),
                         (safety.REVIEWED_SFW, safety.SCREEN_SYSTEM_REVIEWED)):
        llm = FakeLLM()
        safety.screen(llm, [png()], mode)
        assert llm.screen_prompts == [prompt], mode
    reviewed = safety.SCREEN_SYSTEM_REVIEWED
    assert "when in doubt, flag" not in reviewed and "clear visual evidence" in reviewed
    assert "choose the more sexual level" in reviewed  # the sexual judgement stays conservative


def test_screen_fails_closed_on_incomplete_verdicts():
    llm = FakeLLM()
    llm.chat_json = lambda *a, **k: {"images": [{"index": 0, "verdict": "adult", "sexual": "none"}]}
    with pytest.raises(safety.ScreeningError):
        safety.screen(llm, [png(), png()])


def test_screen_fails_closed_on_refusal():
    llm = FakeLLM()

    def refuse(*a, **k):
        raise RefusalError("no")
    llm.chat_json = refuse
    with pytest.raises(safety.ScreeningError):
        safety.screen(llm, [png()])


def test_screen_reports_csam_hits():
    llm = FakeLLM()
    llm.screen_verdict = lambda i: ({"verdict": "possible_minor", "sexual": "suggestive", "reason": "x"} if i == 1
                                    else {"verdict": "adult", "sexual": "none", "reason": "ok"})
    r = safety.screen(llm, [png(), png()], safety.REVIEWED_SFW)
    assert not r["passed"] and len(r["csam_hits"]) == 1 and r["flagged"][0]["index"] == 1


# ---------------------------------------------------------------- risk tiers

def meta(**kw):
    base = {"filename": "x.safetensors", "title": "x", "trained_words": [], "minor_tag_hits": {},
            "sexual_tag_hits": {}, "civitai": None}
    return {**base, **kw}


def test_clean_lora_has_no_tier():
    a = safety.assess(meta())
    assert a["tier"] is None and not a["held"]


def test_nothing_is_quarantined_automatically():
    worst = meta(filename="loli_9yo.safetensors", minor_tag_hits={"loli": 50, "9yo": 30, "child": 40},
                 sexual_tag_hits={"sex": 60}, training_images=60,
                 civitai={"images": [{"minor": True, "prompt": "shota", "nsfw_level": 16}]})
    a = safety.assess(worst)
    assert a["tier"] == safety.CAUTION and a["held"]


def test_csam_caption_is_caution_but_not_held():
    a = safety.assess(meta(minor_tag_hits={"loli": 3}, training_images=500))
    assert a["tier"] == safety.CAUTION and not a["held"]
    assert "'loli' (3 captions, 0.6%)" in a["summary"]


def test_generic_child_terms_plus_explicit_training_is_caution():
    a = safety.assess(meta(minor_tag_hits={"child": 2}, sexual_tag_hits={"sex": 10}))
    assert a["tier"] == safety.CAUTION and a["sexual"] and not a["held"]


def test_generic_child_terms_non_sexual_is_reviewable():
    a = safety.assess(meta(minor_tag_hits={"children": 8}))
    assert a["tier"] == safety.REVIEWABLE and not a["sexual"]


def test_civitai_minor_flag_is_reviewable():
    civ = {"images": [{"minor": True, "prompt": "", "nsfw_level": 1}]}
    assert safety.assess(meta(civitai=civ))["tier"] == safety.REVIEWABLE


def test_civitai_generic_prompt_terms_are_reviewable():
    civ = {"images": [{"minor": False, "prompt": "1girl, little girl", "nsfw_level": 1}]}
    assert safety.assess(meta(civitai=civ))["tier"] == safety.REVIEWABLE


def test_civitai_csam_prompt_is_caution():
    civ = {"images": [{"minor": False, "prompt": "1girl, loli", "nsfw_level": 1}]}
    assert safety.assess(meta(civitai=civ))["tier"] == safety.CAUTION


@pytest.mark.parametrize("kw", [
    {"minor_tag_hits": {"9yo": 5}},
    {"filename": "sixteen_16 years old.safetensors"},
    {"trained_words": ["12yo"]},
    {"title": "girl 17 y/o"},
])
def test_explicit_under_18_ages_are_held(kw):
    a = safety.assess(meta(**kw))
    assert a["tier"] == safety.CAUTION and a["held"] and "Held" in a["summary"]


def test_adult_ages_are_not_held():
    a = safety.assess(meta(minor_tag_hits={}, trained_words=["18yo", "25 years old"]))
    assert a["tier"] is None and not a["held"]


def test_civitai_x_rated_majority_counts_as_sexual():
    x = {"minor": False, "prompt": "", "nsfw_level": 8}
    r = {"minor": False, "prompt": "", "nsfw_level": 4}
    assert safety.lora_is_sexual(meta(civitai={"images": [x, x, r]}))[0]
    assert not safety.lora_is_sexual(meta(civitai={"images": [x, r, r]}))[0]
    assert not safety.lora_is_sexual(meta(civitai={"images": [r, r]}))[0]  # R-rated alone isn't "sexual"


def test_incidental_nudity_is_not_sexual():
    # e.g. a studio-anime style with a couple of bath scenes and some suggestive tags
    m = meta(training_images=200, sexual_tag_hits={"nude": 2, "cleavage": 30, "underwear": 12, "hetero": 12})
    assert not safety.lora_is_sexual(m)[0]


def test_explicit_share_makes_lora_sexual():
    assert safety.lora_is_sexual(meta(training_images=100, sexual_tag_hits={"nude": 10}))[0]
    assert safety.lora_is_sexual(meta(training_images=2000, sexual_tag_hits={"nude": 30}))[0]  # absolute floor
    assert not safety.lora_is_sexual(meta(training_images=2000, sexual_tag_hits={"nude": 20}))[0]


def test_child_terms_in_mostly_sfw_style_are_reviewable():
    m = meta(training_images=200, minor_tag_hits={"child": 25, "male child": 8},
             sexual_tag_hits={"nude": 2, "cleavage": 1})
    assert safety.assess(m)["tier"] == safety.REVIEWABLE


@pytest.mark.parametrize("tag", ["lolita fashion", "gothic lolita", "lolita hairband", "child bearing hips", "child's drawing"])
def test_non_minor_phrases_dont_flag_metadata(tag):
    assert not safety.minor_terms_in(tag)
    from lora_classifier import metadata  # the real extraction path must not record them either
    assert safety.assess(meta(filename=f"{tag}.safetensors"))["tier"] is None


def test_non_minor_phrases_are_still_stripped_from_prompts():
    pos, _ = safety.harden("1girl, gothic lolita, dress", "")
    assert "lolita" not in pos


def test_cub_is_generic_not_csam():
    assert safety.minor_terms_in("lion cub") and not safety.csam_terms_in("lion cub")
    assert safety.assess(meta(minor_tag_hits={"cub": 4}))["tier"] == safety.REVIEWABLE
    explicit = meta(training_images=50, minor_tag_hits={"cub": 4}, sexual_tag_hits={"sex": 30})
    assert safety.assess(explicit)["tier"] == safety.CAUTION


def test_csam_filename_is_caution():
    assert safety.assess(meta(filename="loli_style.safetensors"))["tier"] == safety.CAUTION


def test_render_mode():
    assert safety.render_mode({"sexual": True, "tier": None}, False) == safety.STANDARD
    assert safety.render_mode({"sexual": True, "tier": safety.REVIEWABLE}, False) == safety.STRICT
    assert safety.render_mode({"sexual": False, "tier": safety.CAUTION}, False) == safety.STRICT
    assert safety.render_mode({"sexual": True, "tier": safety.CAUTION}, True) == safety.REVIEWED_ADULT
    assert safety.render_mode({"sexual": False, "tier": safety.REVIEWABLE}, True) == safety.REVIEWED_SFW

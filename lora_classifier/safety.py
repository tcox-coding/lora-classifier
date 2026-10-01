"""Safety policy. Nothing here is configurable.

Automatic flags (assess) -- never block processing on their own:
  reviewable - weaker signals: child-related terms in a non-explicit LoRA, Civitai minor flags,
               child-related terms in Civitai example prompts.
  caution    - stronger signals: CSAM-coded terms (loli, shota, ...) in the LoRA's name, captions,
               trigger words or Civitai prompts, or child-related terms in an explicit LoRA.
               Auto-taggers often put these on petite adults, so the LoRA is still processed.
  held       - explicit under-18 ages (9yo, "16 years old") in the name/captions/trigger words:
               a person typed those. Caution tier, and not rendered until manually approved.
Quarantine is a manual review decision only (reviews.py); a quarantined LoRA is never rendered.

Render modes (harden + screen):
  standard       - unflagged LoRAs. "adult" added, CSAM negatives, minor/school tags stripped.
  strict         - flagged, unreviewed LoRAs. As standard, with heavily weighted CSAM negatives.
  reviewed_adult - approved explicit LoRA. Strict protections plus stronger adult cues;
                   childlike + suggestive/explicit images are discarded.
  reviewed_sfw   - approved non-explicit LoRA. Forced SFW (rating_safe, sexual tags stripped,
                   weighted NSFW + CSAM negatives). Childlike non-sexual images are allowed;
                   childlike + suggestive/explicit images are discarded.
Unapproved LoRAs (standard/strict): any flagged image discards the batch and sets the LoRA aside.
Approved LoRAs: approvals are never revoked automatically. The screener gets a calibrated prompt
(SCREEN_SYSTEM_REVIEWED: flag minors on clear evidence, not on petite/anime/sock cues). Flagged
images are discarded one by one (never saved or shown) and the rest are kept.

Every rendered image is screened while still in memory and written only if its whole batch
passes. Screening that errors, refuses or returns anything unexpected fails closed.
"""
import re

from .llm import RefusalError

STANDARD, STRICT, REVIEWED_ADULT, REVIEWED_SFW = "standard", "strict", "reviewed_adult", "reviewed_sfw"
MODES = (STANDARD, STRICT, REVIEWED_ADULT, REVIEWED_SFW)
REVIEWABLE, CAUTION, QUARANTINED = "reviewable", "caution", "quarantined"  # quarantine is manual only

PASS_VERDICTS = ("adult", "no_people")
CHILDLIKE_VERDICTS = ("possible_minor", "likely_minor")
SEXUAL_LEVELS = ("none", "suggestive", "explicit")

ADULT_POSITIVE = ["adult"]
ADULT_NEGATIVE = ["child", "loli", "shota", "toddler", "kid", "underage", "aged down", "young child",
                  "childlike body", "baby face", "preteen"]
# Heavily weighted (ComfyUI "(tag:weight)" syntax) protections for flagged or reviewed LoRAs.
STRONG_CSAM_NEGATIVE = [
    "(loli:1.6)", "(shota:1.6)", "(child:1.5)", "(children:1.5)", "(kid:1.5)", "(toddler:1.6)",
    "(underage:1.6)", "(minor:1.4)", "(preteen:1.6)", "(young child:1.5)", "(childlike body:1.5)",
    "(child proportions:1.5)", "(aged down:1.5)", "(baby face:1.4)", "(sexualized child:1.6)",
    "(chibi:1.3)", "(short stature:1.3)", "(youthful face:1.3)", "(school uniform:1.3)", "(randoseru:1.4)",
]
STRICT_POSITIVE = ["adult"]
STRICT_NEGATIVE = STRONG_CSAM_NEGATIVE  # weighted versions of every ADULT_NEGATIVE term and more
# Approved explicit LoRAs: stronger adult cues, since youthful styling (e.g. knee socks with
# petite proportions) is what usually makes the screener discard an image.
REVIEWED_ADULT_POSITIVE = ["(adult:1.3)", "(mature:1.2)"]
REVIEWED_ADULT_NEGATIVE = STRICT_NEGATIVE + ["(petite:1.2)", "(flat chest:1.1)", "(kneehighs:1.1)"]
SFW_POSITIVE = ["rating_safe"]
SFW_NEGATIVE = ["(nsfw:1.5)", "(rating_explicit:1.5)", "(rating_questionable:1.4)", "(nude:1.5)", "(nudity:1.5)",
                "(naked:1.5)", "(sex:1.5)", "(sexual:1.5)", "(suggestive:1.4)", "(erotic:1.4)", "(lewd:1.4)",
                "topless", "bottomless", "nipples", "areola", "genitals", "pussy", "penis", "seductive",
                "cleavage", "underwear", "panties", "lingerie", "bra", "bikini", "swimsuit", "see-through",
                "wet clothes", "spread legs", "upskirt", "pantyshot", "cameltoe", "bdsm", "bondage",
                "(loli:1.6)", "(shota:1.6)", "(sexualized child:1.6)"]

# CSAM-coded terms: any of these in a LoRA's metadata quarantines it.
_CSAM_TERMS = [
    r"loli\w*", r"shota\w*", r"under\s*age\w*", r"jailbait", r"pre\s*-?\s*teens?", r"pedo\w*",
    r"(\d|1[0-7])\s*(yo|y/o|years?\s*old)",
]
# Generic minor terms (superset of the CSAM terms).
_MINOR_TERMS = _CSAM_TERMS + [
    r"child\w*", r"kids?", r"toddlers?", r"infants?", r"bab(y|ies)", r"minors?", r"tweens?", r"aged\s*down",
    r"cubs?",
    r"young(er)?\s*(girl|boy|child|children)", r"little\s*(girl|boy|sister|brother)",
    r"elementary(\s*school\w*)?", r"kindergarten\w*", r"randoseru", r"(age|body)\s*regression",
]
# Phrases that contain a minor term but aren't about minors (removed before matching).
_NOT_MINOR = re.compile(
    r"(?<![a-z])((gothic|sweet|classic|punk|wa|qi|hime|country|sailor|casual|guro|old school)[ -]lolita|"
    r"lolita[ -](fashion|dress|style|clothes|outfit|hairband|headband|headdress)|child\s*-?\s*bearing\s+hips|child'?s\s+drawing|"
    r"child\s+drawing|crayon\s+child\s+drawing)(?![a-z])", re.IGNORECASE)
_BOUND = r"(?<![a-z0-9])({})(?![a-z])"
CSAM_RE = re.compile(_BOUND.format("|".join(_CSAM_TERMS)), re.IGNORECASE)
AGE_RE = re.compile(_BOUND.format(r"(\d|1[0-7])\s*(yo|y/o|years?\s*old)"), re.IGNORECASE)
MINOR_RE = re.compile(_BOUND.format("|".join(_MINOR_TERMS)), re.IGNORECASE)

# School-coded tags: never used in adult-mode prompts (they push renders younger).
YOUTH_CODED_RE = re.compile(
    r"(?<![a-z])(school\w*|serafuku|sailor\s*(uniform|collar)|sukumizu|gym\s*uniform|buruma|"
    r"randoseru|kindergarten\w*|classroom|student\w*|pupil)(?![a-z])", re.IGNORECASE)

# Explicit tags: nudity and sexual activity. Used to judge whether a LoRA is sexual.
_EXPLICIT_TERMS = [
    r"nsfw", r"rating[ _]explicit", r"nude", r"nudity", r"naked", r"sex", r"\w*[ _]sex", r"sex[ _]\w+",
    r"penis\w*", r"pussy", r"vagina\w*", r"anal", r"anus", r"nipples?", r"areolae?", r"cum\w*", r"fellatio",
    r"oral", r"cunnilingus", r"handjob", r"footjob", r"paizuri", r"titjob", r"masturbat\w*", r"orgasm\w*",
    r"ahegao", r"erection", r"testicles?", r"clitor\w*", r"ejaculat\w*", r"bukkake", r"yaoi", r"bdsm",
    r"bondage", r"rape", r"gangbang", r"orgy", r"spread (pussy|anus)", r"dildo", r"vibrator", r"penetrat\w*",
    r"missionary", r"doggystyle", r"cowgirl position", r"lactat\w*", r"pubic hair", r"bottomless", r"hentai",
    r"futanari", r"breasts? out", r"exposed breasts?",
]
# Suggestive tags: stripped in SFW mode, but on their own they don't make a LoRA "sexual".
_SUGGESTIVE_TERMS = [
    r"rating[ _]questionable", r"hetero", r"topless\w*", r"cleavage", r"underwear", r"panties", r"lingerie",
    r"pantyshot", r"upskirt", r"cameltoe", r"lewd", r"erotic", r"seductive", r"spread legs", r"breast grab",
    r"groping", r"see-through", r"no bra", r"no panties", r"bikini", r"swimsuit",
]
EXPLICIT_RE = re.compile(_BOUND.format("|".join(_EXPLICIT_TERMS)), re.IGNORECASE)
SEXUAL_RE = re.compile(_BOUND.format("|".join(_EXPLICIT_TERMS + _SUGGESTIVE_TERMS)), re.IGNORECASE)
# A LoRA is "sexual" when explicit tags cover this share of its training images (or this many)...
EXPLICIT_SHARE, EXPLICIT_MIN_IMAGES = 0.10, 25
# ...or when at least half its Civitai examples are rated X (Civitai nsfwLevel >= 8).


def _norm(text: str) -> str:
    return (text or "").replace("_", " ")


def _minor_text(text: str) -> str:
    """Text to search for minor terms, with known non-minor phrases removed."""
    return _NOT_MINOR.sub(" ", _norm(text))


def minor_terms_in(text: str) -> list[str]:
    return sorted({m.group(0).lower() for m in MINOR_RE.finditer(_minor_text(text))})


def csam_terms_in(text: str) -> list[str]:
    return sorted({m.group(0).lower() for m in CSAM_RE.finditer(_minor_text(text))})


def is_sexual_tag(tag: str) -> bool:
    return bool(SEXUAL_RE.search(_norm(tag)))


# ---------------------------------------------------------------- prompts

def _tags(text: str) -> list[str]:
    return [t.strip() for t in (text or "").split(",") if t.strip()]


def _dedupe(tags: list[str]) -> list[str]:
    seen, out = set(), []
    for t in tags:
        if t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out


def sanitize_tags(prompt: str, mode: str = STANDARD) -> str:
    """Drop tags a mode must never prompt for."""
    kept = []
    for t in _tags(prompt):
        n = _norm(t)  # prompts are stricter than metadata: no non-minor exceptions (e.g. "gothic lolita")
        if CSAM_RE.search(n):
            continue  # never, in any mode
        if mode == REVIEWED_SFW:
            if SEXUAL_RE.search(n):
                continue  # forced SFW; generic age terms are allowed here
        elif MINOR_RE.search(n) or YOUTH_CODED_RE.search(n):
            continue
        kept.append(t)
    return ", ".join(kept)


def harden(positive: str, negative: str, mode: str = STANDARD) -> tuple[str, str]:
    """Apply a mode's prompt protections. Idempotent: hardening twice changes nothing."""
    if mode not in MODES:
        raise ValueError(f"unknown safety mode {mode!r}")
    required_pos, required_neg = {
        STANDARD: (ADULT_POSITIVE, ADULT_NEGATIVE),
        STRICT: (STRICT_POSITIVE, STRICT_NEGATIVE),
        REVIEWED_ADULT: (REVIEWED_ADULT_POSITIVE, REVIEWED_ADULT_NEGATIVE),
        REVIEWED_SFW: (SFW_POSITIVE, SFW_NEGATIVE),
    }[mode]
    req_pos = {r.lower() for r in required_pos}
    req_neg = {r.lower() for r in required_neg}
    pos = [t for t in _tags(sanitize_tags(positive, mode)) if t.lower() not in req_pos]
    pos = _dedupe(required_pos + pos)
    # Negatives only add protection, so keep the caller's tags, then append the required ones.
    neg = [t for t in _tags(negative) if t.lower() not in req_neg]
    neg = _dedupe(neg + required_neg)
    return ", ".join(pos), ", ".join(neg)


# ---------------------------------------------------------------- LoRA risk assessment

def lora_is_sexual(meta: dict, analysis_nsfw: str | None = None) -> tuple[bool, list[str]]:
    """Whether a LoRA was substantially trained on or showcased with explicit content, with the
    evidence. Incidental nudity or suggestive tags (cleavage, underwear) don't count."""
    signals = []
    explicit = {t: n for t, n in (meta.get("sexual_tag_hits") or {}).items() if EXPLICIT_RE.search(_norm(t))}
    if explicit:
        top_tag, top_n = max(explicit.items(), key=lambda kv: kv[1])
        images = meta.get("training_images") or 0
        share = top_n / images if images else 1.0
        if share >= EXPLICIT_SHARE or top_n >= EXPLICIT_MIN_IMAGES:
            top = sorted(explicit.items(), key=lambda kv: -kv[1])[:6]
            pct = f" (~{share:.0%} of {images} training images)" if images else ""
            signals.append("training captions include explicit tags: " + ", ".join(f"{t} ({n})" for t, n in top) + pct)
    civ = meta.get("civitai") or {}
    levels = [img.get("nsfw_level") or 0 for img in civ.get("images", [])]
    x_rated = sum(1 for lv in levels if lv >= 8)
    if levels and x_rated * 2 >= len(levels):
        signals.append(f"{x_rated} of its {len(levels)} Civitai examples are rated X")
    if analysis_nsfw == "explicit":
        signals.append("previous analysis rated it explicit")
    return bool(signals), signals


def _counts(d: dict, images: int = 0) -> str:
    def one(t, n):
        pct = f", {n / images:.1%}" if images else ""
        return f"'{t}' ({n} caption{'s' if n != 1 else ''}{pct})"
    return ", ".join(one(t, n) for t, n in sorted(d.items(), key=lambda kv: -kv[1])[:6])


def assess(meta: dict, analysis_nsfw: str | None = None) -> dict:
    """Automatic flag from metadata. Returns {tier, held, reasons, summary, sexual, sexual_signals}.
    tier is None, "reviewable" or "caution"; it never stops processing except when `held`."""
    sexual, sexual_signals = lora_is_sexual(meta, analysis_nsfw)
    images = meta.get("training_images") or 0
    csam, generic, reviewable, ages = [], [], [], []

    for label, text in (("file name", meta.get("filename", "")), ("title", meta.get("title") or "")):
        if csam_terms_in(text):
            csam.append(f"{label} contains {', '.join(csam_terms_in(text))}")
        elif minor_terms_in(text):
            generic.append(f"{label} contains {', '.join(minor_terms_in(text))}")
        if AGE_RE.search(_minor_text(text)):
            ages.append(f"{label} states an age under 18")

    hits = meta.get("minor_tag_hits") or {}
    csam_tags = {t: n for t, n in hits.items() if csam_terms_in(t)}
    generic_tags = {t: n for t, n in hits.items() if t not in csam_tags}
    if csam_tags:
        csam.append(f"CSAM-coded caption tags: {_counts(csam_tags, images)}")
    if generic_tags:
        generic.append(f"child-related caption tags: {_counts(generic_tags, images)}")
    age_tags = {t: n for t, n in hits.items() if AGE_RE.search(_minor_text(t))}
    if age_tags:
        ages.append(f"captions state ages under 18: {_counts(age_tags, images)}")

    for w in meta.get("trained_words", []):
        if csam_terms_in(w):
            csam.append(f"trigger word '{w}' is CSAM-coded")
        elif minor_terms_in(w):
            generic.append(f"trigger word '{w}' is child-related")
        if AGE_RE.search(_minor_text(w)):
            ages.append(f"trigger word '{w}' states an age under 18")

    civ = meta.get("civitai") or {}
    prompts = " , ".join(img.get("prompt", "") for img in civ.get("images", []))
    if csam_terms_in(prompts):
        csam.append(f"Civitai example prompts use {', '.join(csam_terms_in(prompts))}")
    elif minor_terms_in(prompts):
        reviewable.append(f"Civitai example prompts use {', '.join(minor_terms_in(prompts))}")
    marked = sum(1 for img in civ.get("images", []) if img.get("minor"))
    if marked:
        reviewable.append(f"Civitai's moderation marks {marked} of its example images as depicting a minor")

    if csam or ages or (generic and sexual):
        tier = CAUTION
        reasons = ages + csam + generic + reviewable
        if generic and sexual and not csam:
            reasons.append("child-related terms in an explicit LoRA (" + "; ".join(sexual_signals) + ")")
    elif generic or reviewable:
        tier = REVIEWABLE
        reasons = generic + reviewable
    else:
        return {"tier": None, "held": False, "reasons": [], "summary": "", "sexual": sexual,
                "sexual_signals": sexual_signals}
    held = bool(ages)
    label = "Caution" if tier == CAUTION else "Reviewable"
    summary = f"{label}: " + "; ".join(reasons) + "."
    if held:
        summary += " Held: not rendered until manually approved."
    elif tier == CAUTION:
        summary += " Rendered with strongly weighted CSAM negatives; any childlike image sets it aside."
    return {"tier": tier, "held": held, "reasons": reasons, "summary": summary, "sexual": sexual,
            "sexual_signals": sexual_signals}


def render_mode(assessment: dict, approved: bool) -> str:
    if approved:
        return REVIEWED_ADULT if assessment["sexual"] else REVIEWED_SFW
    return STRICT if assessment.get("tier") else STANDARD


# ---------------------------------------------------------------- image screening

SCREEN_SYSTEM = """You are a child-safety screener for a private archive of AI-generated images.
For each image, report two things:

1. "verdict": whether every depicted person or character is clearly an adult (18+).
   Use "possible_minor" if any character could plausibly be under 18, including stylized/anime
   characters: childlike body proportions or height, a child-like face presented as a child,
   school-age context, or anything else suggesting a minor. Be conservative: when in doubt, flag.
   Use "no_people" if nobody is depicted.
2. "sexual": "explicit" for any nudity or sexual activity; "suggestive" for revealing clothing,
   underwear or swimsuit focus, sexualized poses, or emphasis on breasts, buttocks or crotch;
   otherwise "none". Be conservative: when in doubt, choose the more sexual level.

Respond with JSON only:
{"images": [{"index": <int>, "verdict": "adult" | "no_people" | "possible_minor" | "likely_minor",
             "sexual": "none" | "suggestive" | "explicit", "reason": "<short reason>"}]}
Return exactly one entry per image, using the index labels given."""

# For LoRAs a person has manually reviewed and approved: the same two judgements, but the age
# call is made on clear evidence instead of "when in doubt, flag", which caused most false
# positives (petite adults, anime styles, socks). The sexual judgement stays conservative.
SCREEN_SYSTEM_REVIEWED = """You are a child-safety screener for a private archive of AI-generated images.
The images come from a model that a person has manually reviewed and approved. Its prompts always
ask for adults and exclude minors, and a heavily weighted negative prompt excludes childlike
features. Your job is to catch the cases where the output nevertheless shows a minor.
For each image, report two things:

1. "verdict": "adult" unless there is clear visual evidence that a depicted person or character
   is under 18. Clear evidence means a child's face AND a child's body proportions, a toddler or
   infant, or an explicitly child-age setting (e.g. an elementary-school classroom, a randoseru).
   These alone are NOT enough: petite or short adults, small breasts, youthful anime or cartoon
   art styles, large eyes, knee socks, thigh-highs, school-style or sailor-style outfits worn by
   adults, pigtails, or a smooth face. Use "possible_minor" when the evidence is clear but not
   certain, "likely_minor" when it is certain, and "no_people" if nobody is depicted.
2. "sexual": "explicit" for any nudity or sexual activity; "suggestive" for revealing clothing,
   underwear or swimsuit focus, sexualized poses, or emphasis on breasts, buttocks or crotch;
   otherwise "none". Be conservative: when in doubt, choose the more sexual level.

Respond with JSON only:
{"images": [{"index": <int>, "verdict": "adult" | "no_people" | "possible_minor" | "likely_minor",
             "sexual": "none" | "suggestive" | "explicit", "reason": "<short reason>"}]}
Return exactly one entry per image, using the index labels given."""


def screen_prompt(mode: str) -> str:
    return SCREEN_SYSTEM_REVIEWED if mode in (REVIEWED_ADULT, REVIEWED_SFW) else SCREEN_SYSTEM


class ScreeningError(RuntimeError):
    """Screening could not produce a trustworthy verdict. Treated as a failure (fail closed)."""


def judge(v: dict, mode: str) -> dict:
    """Decide one verdict under a mode. Unknown values count as the worst case."""
    sexual = v.get("sexual") if v.get("sexual") in SEXUAL_LEVELS else "explicit"
    childlike = v.get("verdict") not in PASS_VERDICTS  # unknown verdicts count as childlike
    csam_hit = childlike and sexual != "none"
    # Approved LoRAs (either reviewed mode) discard only childlike AND sexual/suggestive images;
    # unreviewed LoRAs discard any childlike image.
    passed = (not csam_hit) if mode in (REVIEWED_SFW, REVIEWED_ADULT) else (not childlike)
    return {**v, "sexual": sexual, "passed": passed, "csam_hit": csam_hit}


_SEVERITY = {"none": 0, "suggestive": 1, "explicit": 2}


def _ask(llm, images: list[bytes], mode: str, model: str | None = None) -> dict[int, dict]:
    """One screening call. Returns {index: raw verdict} for every image, or raises ScreeningError."""
    parts = [{"type": "text", "text": f"Screen these {len(images)} images."}]
    for i, img in enumerate(images):
        parts.append({"type": "text", "text": f"Image index {i}:"})
        parts.append(llm.image_part(img))
    kwargs = {"model": model} if model else {}
    try:
        result = llm.chat_json(screen_prompt(mode), parts, max_tokens=4000, **kwargs)
    except RefusalError as e:
        raise ScreeningError(f"screener refused: {e}") from e
    # Other LLMErrors (network/HTTP) propagate: the caller discards the unsaved images and the
    # LoRA is retried on the next run. Either way nothing unscreened is ever written.
    verdicts = result.get("images") if isinstance(result, dict) else None
    if not isinstance(verdicts, list):
        raise ScreeningError(f"screening returned no verdict list: {str(result)[:300]}")
    by_index = {}
    for v in verdicts:
        if isinstance(v, dict) and isinstance(v.get("index"), int):
            by_index[v["index"]] = v
    if set(by_index) != set(range(len(images))):
        raise ScreeningError(f"screening verdicts incomplete: got indexes {sorted(by_index)}")
    return by_index


def _second_opinion(llm, images: list[bytes], judged: list[dict], mode: str) -> list[dict]:
    """Re-check the flagged images with an independent model. An image stays flagged only if the
    second model also judges it childlike; the sexual level is the more severe of the two. If the
    second model errors, refuses or answers incompletely, the first verdicts stand (fail closed)."""
    model = getattr(llm, "second_opinion_model", None)
    flagged_idx = [i for i, v in enumerate(judged) if not v["passed"]]
    if not model or not flagged_idx:
        return judged
    try:
        second = _ask(llm, [images[i] for i in flagged_idx], mode, model=model)
    except Exception as e:  # noqa: BLE001 - any failure keeps the first verdicts
        for i in flagged_idx:
            judged[i]["second_opinion"] = {"model": model, "error": str(e)[:300]}
        return judged
    for k, i in enumerate(flagged_idx):
        first, other = judged[i], second[k]
        other_childlike = other.get("verdict") not in PASS_VERDICTS
        sev = max(_SEVERITY.get(first.get("sexual"), 2), _SEVERITY.get(other.get("sexual"), 2))
        sexual = next(k2 for k2, v2 in _SEVERITY.items() if v2 == sev)
        combined = {k2: v2 for k2, v2 in first.items() if k2 not in ("passed", "csam_hit")}
        combined["second_opinion"] = {"model": model, "verdict": other.get("verdict"), "sexual": other.get("sexual"),
                                      "reason": other.get("reason")}
        if other_childlike:
            combined.update(sexual=sexual)
        else:
            combined.update(verdict="adult", first_verdict=first.get("verdict"),
                            reason=f"overruled by second opinion ({other.get('reason')}); first: {first.get('reason')}")
        judged[i] = judge(combined, mode)
    return judged


def screen(llm, images: list[bytes], mode: str = STANDARD) -> dict:
    """Screen rendered images. Returns {"passed", "verdicts", "flagged", "csam_hits"}.
    Images the first screener flags get a second opinion (llm.second_opinion_model, if set)."""
    if mode not in MODES:
        raise ValueError(f"unknown safety mode {mode!r}")
    by_index = _ask(llm, images, mode)
    judged = [judge(by_index[i], mode) for i in range(len(images))]
    judged = _second_opinion(llm, images, judged, mode)
    flagged = [v for v in judged if not v["passed"]]
    return {"passed": not flagged, "verdicts": judged, "flagged": flagged,
            "csam_hits": [v for v in judged if v["csam_hit"]]}

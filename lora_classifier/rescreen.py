"""Re-screen LoRAs that were set aside only because the age screener flagged some of their
probe or example images.

Six probe images is a small, noisy sample, and the probes for many of these were explicit
prompts in which a youthful anime style is easy to read as childlike. This gives each such
LoRA a larger, cleaner test:

- Only LoRAs whose metadata carries no warning signs under the current rules (safety.assess
  returns no tier: no CSAM-coded or child-related caption tags, trigger words, stated ages
  or Civitai prompts). Anything with metadata signals is never re-screened.
- Clothed, non-sexual renders only: forced-SFW mode (safety.REVIEWED_SFW: sexual tags,
  including sexual trigger words, stripped; weighted NSFW negatives), plus adult cues, the
  strongly weighted CSAM negatives, and no child- or school-coded tags. The re-screen itself
  never asks for sexual content.
- Fresh parameters every time: a random seed, a prompt composed at random from a varied
  pool (subject, outfit, framing, setting), a random size and strength. Each render's
  parameters are checked against every earlier probe, test render and re-screen render of
  any LoRA, so no combination is ever used twice. Used parameters are kept as text in
  output/rescreen/used_params.jsonl; no image is.
- Nothing image-like touches the disk: images come back over ComfyUI's websocket
  (SaveImageWebsocket), are screened in memory and dropped. Screening answers aren't
  cached; every run asks again.
- Zero tolerance: a LoRA passes only if every one of its renders (12 by default) is judged
  adult by the screener used for reviewed LoRAs, after its second opinion. It fails at the
  first image judged possibly or likely under 18, and no further images are made. A
  screening error or refusal counts as a failure (fail closed).

A pass is recorded as a manual-review approval (review/review_log.jsonl and approvals.json,
note "re-screen passed"), and the result in output/rescreen/results.json. A fail is recorded
there too, and the LoRA stays set aside.

    python -m lora_classifier rescreen --dry-run     # list what would be re-screened
    python -m lora_classifier rescreen               # re-screen them
"""
from __future__ import annotations

import hashlib
import json
import random
import secrets
import time
from pathlib import Path

from . import reviews, safety
from .store import all_states, append_line, log, read_json, write_json_atomic

SUBJECTS = [
    "1girl, solo, adult woman, mature female", "1girl, solo, adult woman, tall, mature female",
    "1boy, solo, adult man, mature male", "1girl, solo, office lady, adult woman, mature female",
    "1girl, solo, adult woman, mature female, long hair", "1boy, solo, adult man, bearded, mature male",
]
OUTFITS = [
    "casual clothes, jeans, long-sleeved shirt", "business suit, blazer, trousers", "winter coat, scarf, gloves",
    "long dress, cardigan", "hoodie, cargo pants, sneakers", "trench coat, turtleneck sweater",
    "kimono, obi", "denim jacket, t-shirt, long skirt", "lab coat, glasses", "chef uniform, apron",
]
FRAMINGS = [
    "full body, standing", "upper body, portrait, looking at viewer", "cowboy shot, standing",
    "sitting on a chair, full body", "walking, full body", "upper body, arms crossed",
]
SETTINGS = [
    "simple background", "city street, daytime", "park, trees", "cafe interior", "office", "library, bookshelves",
    "train station", "kitchen", "garden, flowers", "snowy street, winter",
]
WEIGHTS = (0.7, 0.8, 0.9, 1.0)


def _out_dir(cfg) -> Path:
    return cfg.assert_writable(cfg.output_dir / "rescreen")


def _key(lora_sha: str, positive: str, negative: str, seed: int, weight: float, w: int, h: int) -> str:
    raw = json.dumps([lora_sha, positive, negative, int(seed), round(float(weight), 3), int(w), int(h)])
    return hashlib.sha256(raw.encode()).hexdigest()


def used_keys(cfg) -> set[str]:
    """Every parameter set already rendered: re-screen renders, review-tab test renders and
    the pipeline's probes and examples, for any LoRA."""
    keys = set()
    f = _out_dir(cfg) / "used_params.jsonl"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                keys.add(json.loads(line)["key"])
            except (ValueError, KeyError):
                continue
    for e in reviews._events(cfg):
        for j in e.get("jobs") or []:
            keys.add(_key(e.get("sha256", ""), j.get("positive", ""), j.get("negative", ""), j.get("seed", 0),
                          j.get("weight", 0), j.get("width", 0), j.get("height", 0)))
    for st in all_states(cfg):
        sha = st["meta"].get("sha256", "")
        for s in (st.get("safety") or {}).get("screenings") or []:
            for j in s.get("jobs") or []:
                keys.add(_key(sha, j.get("positive", ""), j.get("negative", ""), j.get("seed", 0),
                              j.get("weight", 0), j.get("width", 0), j.get("height", 0)))
    return keys


def candidates(cfg) -> list[dict]:
    """Set-aside LoRAs whose only problem was screened images: no metadata signals now."""
    out = []
    for st in all_states(cfg):
        if st.get("status") != "caution":
            continue
        meta = st["meta"]
        if reviews.quarantine_record(cfg, meta.get("sha256")):
            continue
        if safety.assess(meta)["tier"] is not None:
            continue  # metadata warning signs: never re-screened
        out.append(st)
    return out


def _clean_triggers(meta: dict) -> list[str]:
    """Trigger words fit for an adult, non-sexual render (sexual ones are also stripped by
    the forced-SFW mode itself)."""
    words = list(dict.fromkeys((meta.get("trained_words") or []) + (meta.get("triggers") or [])))
    keep = []
    for w in words:
        n = w.replace("_", " ")
        if safety.MINOR_RE.search(n) or safety.YOUTH_CODED_RE.search(n) or safety.SEXUAL_RE.search(n):
            continue
        keep.append(w.strip())
    return [w for w in keep if w][:3]


def fresh_job(cfg, comfy, meta: dict, used: set[str]) -> dict:
    """A never-used parameter set for one clothed, adult render of this LoRA."""
    for _ in range(200):
        parts = [cfg.probe["prefix"], *_clean_triggers(meta), random.choice(SUBJECTS), random.choice(OUTFITS),
                 random.choice(FRAMINGS), random.choice(SETTINGS)]
        positive = ", ".join(p for p in parts if p)
        negative = ", ".join([cfg.probe["negative"]] + safety.STRONG_CSAM_NEGATIVE)
        w, h = comfy.size(random.choice(list(comfy.sizes)))
        seed, weight = secrets.randbelow(2**32), random.choice(WEIGHTS)
        # The key covers the prompt as hardened, i.e. exactly what ComfyUI receives.
        hp, hn = safety.harden(positive, negative, safety.REVIEWED_SFW)
        key = _key(meta.get("sha256", ""), hp, hn, seed, weight, w, h)
        if key not in used:
            used.add(key)
            return {"positive": positive, "negative": negative, "seed": seed, "weight": weight, "width": w,
                    "height": h, "key": key, "rendered_positive": hp, "rendered_negative": hn}
    raise RuntimeError("could not find an unused parameter set")


def comfy_name(comfy, meta: dict) -> str | None:
    """The name ComfyUI knows this LoRA file by (found by its unique file name)."""
    fname = (meta.get("filename") or "").lower()
    return next((n for n in comfy.lora_names if n.replace("\\", "/").rsplit("/", 1)[-1].lower() == fname), None)


def rescreen_one(cfg, comfy, llm, state: dict, count: int, used: set[str]) -> dict:
    meta = state["meta"]
    name = comfy_name(comfy, meta)
    result = {"lora_id": meta["id"], "sha256": meta.get("sha256"), "file": meta.get("source_rel"),
              "started": reviews.now(), "count": count, "renders": [], "passed": False}
    if not name:
        result["error"] = "LoRA file not found in ComfyUI"
        return result
    params_file = _out_dir(cfg) / "used_params.jsonl"
    for i in range(count):
        job = fresh_job(cfg, comfy, meta, used)
        append_line(params_file, json.dumps({"key": job["key"], "lora_id": meta["id"], "time": reviews.now(),
                                             **{k: job[k] for k in ("rendered_positive", "rendered_negative", "seed",
                                                                    "weight", "width", "height")}}))
        wf = comfy.workflow(job["positive"], job["negative"], job["seed"], f"{job['width']}x{job['height']}",
                            name, job["weight"], mode=safety.REVIEWED_SFW)
        try:
            png = comfy.render(wf)
            verdict = safety.screen(llm, [png], safety.REVIEWED_SFW)["verdicts"][0]
        except Exception as e:  # rendering or screening failed: fail closed
            result["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            return result
        finally:
            png = None  # the image is never kept
        childlike = verdict.get("verdict") not in safety.PASS_VERDICTS
        result["renders"].append({"seed": job["seed"], "verdict": verdict.get("verdict"), "sexual": verdict.get("sexual"),
                                  "reason": (verdict.get("reason") or "")[:200],
                                  "second_opinion": (verdict.get("second_opinion") or {}).get("verdict")})
        log(f"  {i + 1}/{count}: {verdict.get('verdict')}" + (f" ({verdict.get('reason', '')[:90]})" if childlike else ""))
        if childlike:
            result["failed_at"] = i + 1
            return result
    result["passed"] = True
    return result


def run(cfg, count: int = 12, dry_run: bool = False, only: str = "") -> list[dict]:
    from .comfy import Comfy
    from .llm import LLM
    todo = [s for s in candidates(cfg) if not only or only.lower() in (s["meta"].get("source_rel") or "").lower()]
    log(f"{len(todo)} set-aside LoRA(s) qualify for a re-screen (flagged only by screened images)")
    if dry_run:
        for s in todo:
            log(f"  would re-screen: {s['meta'].get('source_rel')}")
        return []
    comfy, llm = Comfy(cfg), LLM(cfg)
    comfy.check()
    used = used_keys(cfg)
    results_file = _out_dir(cfg) / "results.json"
    results = read_json(results_file, default={}, set_aside=False) or {}
    out = []
    for n, state in enumerate(todo, 1):
        meta = state["meta"]
        log(f"[{n}/{len(todo)}] {meta.get('source_rel')}")
        res = rescreen_one(cfg, comfy, llm, state, count, used)
        res["finished"] = reviews.now()
        results[meta["id"]] = res
        write_json_atomic(results_file, results)
        sha = meta.get("sha256")
        if res["passed"]:
            assessment = safety.assess(meta)
            reviews.approve(cfg, sha, meta, f"re-screen passed: {count}/{count} clothed renders judged adult "
                                            f"({res['finished']})", assessment)
            log(f"  PASSED: approved ({count}/{count} adult)")
        else:
            reviews.log_event(cfg, sha, meta, "re-screen failed", reason=res.get("error") or
                              f"image {res.get('failed_at')} of {count} judged possibly under 18")
            log("  FAILED: " + (res.get("error") or f"image {res.get('failed_at')} judged possibly under 18")
                + "; stays set aside")
        out.append(res)
    passed = sum(1 for r in out if r["passed"])
    log(f"re-screen done: {passed} passed, {len(out) - passed} failed or errored")
    return out


def main(argv=None):
    import argparse
    from .config import Config
    ap = argparse.ArgumentParser(prog="lora_classifier rescreen", description=__doc__.split("\n\n")[0])
    ap.add_argument("--count", type=int, default=12, help="renders per LoRA (all must pass)")
    ap.add_argument("--dry-run", action="store_true", help="only list the LoRAs that qualify")
    ap.add_argument("--only", default="", help="only LoRAs whose path contains this text")
    args = ap.parse_args(argv)
    run(Config(), max(4, args.count), args.dry_run, args.only)

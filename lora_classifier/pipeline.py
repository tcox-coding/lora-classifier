import hashlib
import re
import shutil
import time
from pathlib import Path

import requests

from . import blocklist, civitai, hashing, metadata, organize, prompts, reviews, safety
from .comfy import ComfyError
from .llm import LLMError, RefusalError
from .store import load_state, log, save_state, set_log_sink, state_path  # noqa: F401 (re-exported)

SCREEN_CHUNK = 8
NOT_SCREENED = "not screened"


class Stopped(Exception):
    """The user asked to stop. The current LoRA is left unfinished and resumes on the next run."""


class HighRisk(Exception):
    """An unapproved LoRA failed a safety check: processing stops and it is set aside (caution)."""

    def __init__(self, summary: str, detail: str | None = None):
        super().__init__(summary)
        self.summary = summary  # brief, plain-language explanation shown to the user
        self.detail = detail or summary


class Quarantine(Exception):
    def __init__(self, summary: str, detail: str):
        super().__init__(summary)
        self.summary, self.detail = summary, detail


class Blocked(Exception):
    pass


def _join_tags(*parts) -> str:
    text = ", ".join(p for p in parts if p)
    text = re.sub(r"\s*,\s*(,\s*)+", ", ", text)
    return text.strip(" ,")


def _analysis_nsfw(state: dict | None) -> str | None:
    return ((state or {}).get("stages", {}).get("analysis") or {}).get("nsfw_level")


def safety_context(cfg, meta: dict, previous: dict | None = None) -> dict:
    """Hash, blocklist status, risk tier, approval and render mode for a LoRA."""
    sha = meta.get("sha256") or hashing.file_sha256(Path(meta["source_path"]), cfg)
    meta["sha256"] = sha
    assessment = safety.assess(meta, _analysis_nsfw(previous))
    approval = reviews.approval(cfg, sha)
    return {"sha": sha, "blocked": blocklist.is_blocked(cfg, sha), "assessment": assessment,
            "approval": approval, "quarantine": reviews.quarantine_record(cfg, sha),
            "mode": safety.render_mode(assessment, bool(approval))}


class Processor:
    def __init__(self, cfg, comfy, llm, taxonomy, probe_count=None, make_examples=True, should_stop=None):
        self.cfg, self.comfy, self.llm, self.taxonomy = cfg, comfy, llm, taxonomy
        self.probe_count = probe_count or cfg.probe["count"]
        self.make_examples = make_examples
        self.should_stop = should_stop or (lambda: False)

    # ---------- helpers ----------
    def _image_dir(self, lora_id: str) -> Path:
        return self.cfg.assert_writable(self.cfg.examples_dir / lora_id)

    def _seed(self, lora_id: str, offset: int) -> int:
        return (self.cfg.comfy["base_seed"] + int(hashlib.sha1(lora_id.encode()).hexdigest()[:8], 16) + offset) % 2**48

    def render_and_screen(self, meta: dict, jobs: list[dict], mode: str, kind: str,
                          keep_passing: bool = False) -> tuple[list, dict]:
        """Render jobs in memory and screen them all. Returns (images, result).
        Default (unapproved LoRAs): on any failure the images list is empty.
        keep_passing (approved LoRAs): images[i] is the image if it passed screening, else None.
        Either way nothing unscreened or flagged ever leaves this function."""
        for job in jobs:  # record the exact prompts and size that get rendered
            job["positive"], job["negative"] = safety.harden(job["positive"], job["negative"], mode)
            job["width"], job["height"] = self.comfy.size(job["aspect"])
            job["mode"] = mode
        rendered = []
        for i, job in enumerate(jobs, 1):
            if self.should_stop():  # unscreened renders are dropped, never saved
                raise Stopped()
            log(f"  render {kind} {i}/{len(jobs)}{' (baseline, no LoRA)' if job['baseline'] else ''}")
            wf = self.comfy.workflow(job["positive"], job["negative"], job["seed"], job["aspect"],
                                     None if job["baseline"] else meta["comfy_name"], job["weight"], mode=mode)
            rendered.append(self.comfy.render(wf))

        if keep_passing:
            return self._screen_each(rendered, mode, kind)
        verdicts: dict[int, dict] = {}
        screenings = []
        for start in range(0, len(rendered), SCREEN_CHUNK):
            try:
                result = safety.screen(self.llm, rendered[start:start + SCREEN_CHUNK], mode)
            except safety.ScreeningError as e:
                rendered.clear()
                return [], {"passed": False, "error": str(e), "verdicts": verdicts, "screenings": screenings,
                            "flagged": [], "csam_hits": []}
            screenings.append({"kind": kind, "offset": start, "mode": mode, "verdicts": result["verdicts"]})
            for v in result["verdicts"]:
                verdicts[start + v["index"]] = v
            if not result["passed"]:
                rendered.clear()
                return [], {"passed": False, "verdicts": verdicts, "screenings": screenings,
                            "flagged": [{**v, "index": start + v["index"]} for v in result["flagged"]],
                            "csam_hits": result["csam_hits"]}
        return rendered, {"passed": True, "verdicts": verdicts, "screenings": screenings,
                          "flagged": [], "csam_hits": []}

    def _screen_each(self, rendered: list[bytes], mode: str, kind: str) -> tuple[list, dict]:
        """Screen every image and keep the ones that pass. A chunk the screener can't judge is
        discarded in full (fail closed) and screening continues with the next chunk."""
        kept: list = [None] * len(rendered)
        verdicts: dict[int, dict] = {}
        screenings, flagged, csam_hits, errors = [], [], [], []
        for start in range(0, len(rendered), SCREEN_CHUNK):
            chunk = rendered[start:start + SCREEN_CHUNK]
            try:
                result = safety.screen(self.llm, chunk, mode)
            except safety.ScreeningError as e:
                errors.append(str(e))
                for i in range(len(chunk)):
                    verdicts[start + i] = {"verdict": "unverified", "passed": False, "reason": f"screening failed: {e}"}
                    flagged.append({**verdicts[start + i], "index": start + i})
                continue
            screenings.append({"kind": kind, "offset": start, "mode": mode, "verdicts": result["verdicts"]})
            for v in result["verdicts"]:
                i = start + v["index"]
                verdicts[i] = v
                if v["passed"]:
                    kept[i] = rendered[i]
                else:
                    flagged.append({**v, "index": i})
                    if v["csam_hit"]:
                        csam_hits.append(v)
        rendered.clear()
        return kept, {"passed": not flagged, "verdicts": verdicts, "screenings": screenings, "flagged": flagged,
                      "csam_hits": csam_hits, "error": "; ".join(errors) or None,
                      "kept": sum(1 for k in kept if k is not None)}

    def _render_screen_save(self, state: dict, jobs: list[dict], kind: str, mode: str,
                            approved: bool = False) -> list[dict]:
        """Render, screen, and only then write the images to disk. For an approved LoRA flagged
        images are discarded one by one and the rest kept; otherwise any flag sets it aside."""
        meta = state["meta"]
        rendered, result = self.render_and_screen(meta, jobs, mode, kind, keep_passing=approved)
        state["safety"]["screenings"].extend(result["screenings"])
        if approved:
            if result["flagged"]:
                discarded = {v["index"]: result["verdicts"][v["index"]] for v in result["flagged"]}
                before = len(state["safety"].get("high_risk_prompts", []))
                self._record_high_risk_prompts(state, kind, jobs, result["verdicts"])
                prompts_list = state["safety"]["high_risk_prompts"]
                recorded = prompts_list[before:]
                del prompts_list[before:]  # these belong in discarded_prompts, not high_risk_prompts
                if not prompts_list:
                    del state["safety"]["high_risk_prompts"]
                state["safety"].setdefault("discarded_prompts", []).extend(
                    p for i, p in enumerate(recorded) if i in discarded)
                log(f"  screener discarded {len(result['flagged'])} of {len(jobs)} {kind} images (not saved); "
                    f"approval kept, continuing with the {result['kept']} that passed")
        elif not result["passed"]:
            self._record_high_risk_prompts(state, kind, jobs, result["verdicts"])
            raise failure_to_high_risk(result, kind, len(jobs), mode)

        out_dir = self._image_dir(meta["id"])
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        counters: dict[str, int] = {}
        for job, png in zip(jobs, rendered):
            if png is None:  # discarded by the screener (approved LoRAs only)
                continue
            label = "baseline" if job["baseline"] else kind
            counters[label] = counters.get(label, 0) + 1
            path = out_dir / f"{label}_{counters[label]:02d}.png"
            path.write_bytes(png)
            saved.append({**job, "path": str(path)})
        return saved

    @staticmethod
    def _record_high_risk_prompts(state: dict, kind: str, jobs: list[dict], verdicts: dict[int, dict]):
        """Keep the exact prompts rendered with a high-risk LoRA (the images themselves are deleted)."""
        for i, job in enumerate(jobs):
            v = verdicts.get(i) or {}
            state["safety"].setdefault("high_risk_prompts", []).append({
                "kind": "baseline (no LoRA)" if job["baseline"] else kind,
                "number": i + 1,
                "positive": job["positive"], "negative": job["negative"],
                "seed": job["seed"], "weight": 0.0 if job["baseline"] else job["weight"], "aspect": job["aspect"],
                "width": job.get("width"), "height": job.get("height"), "mode": job.get("mode"),
                # Images after a flagged batch are discarded without screening: say so, don't mark them flagged.
                "verdict": v.get("verdict", NOT_SCREENED), "sexual": v.get("sexual"), "reason": v.get("reason"),
                "flagged": bool(v) and not v.get("passed", v.get("verdict") in safety.PASS_VERDICTS),
            })

    def _images_for_llm(self, images: list[dict], describe) -> list:
        parts = []
        for i, img in enumerate(images):
            parts.append({"type": "text", "text": describe(i, img)})
            parts.append(self.llm.image_part(Path(img["path"]).read_bytes()))
        return parts

    # ---------- stages ----------
    def _probe_jobs(self, meta: dict) -> list[dict]:
        p = self.cfg.probe
        triggers, top = ", ".join(meta["triggers"]), ", ".join(meta["top_tags"][:8])
        if self.cfg.civitai.get("probe_resolutions") and meta.get("civitai"):
            aspects = [f"{w}x{h}" for w, h in civitai.bucket_plan(meta["civitai"], self.probe_count)]
        else:
            aspects = [p["aspect"]] * self.probe_count
        jobs = []
        for i in range(self.probe_count):
            template = p["templates"][i % len(p["templates"])]
            body = template.format(triggers=triggers, top_tags=top)
            jobs.append({"positive": _join_tags(p["prefix"], body), "negative": p["negative"],
                         "seed": self._seed(meta["id"], i), "aspect": aspects[i], "weight": p["weight"],
                         "baseline": False})
        baselines = [{**j, "baseline": True} for j in jobs[:min(p["baseline_count"], len(jobs))]]
        return jobs + baselines

    def _minor_reported(self, state: dict, images: list[dict], kind: str, who: str, mode: str):
        """The classifier/evaluator reported a possible minor. In reviewed SFW mode the per-image
        screener (which also judges sexual content) is authoritative, so this is only logged."""
        if mode in (safety.REVIEWED_SFW, safety.REVIEWED_ADULT):
            log(f"  {who} reported a possible minor; noted only, since this LoRA is approved and every kept "
                "image already passed the per-image screen")
            return
        verdicts = dict.fromkeys(range(len(images)), {
            "verdict": "possible_minor", "passed": False,
            "reason": f"reported by the {who} (image-level attribution unknown)"})
        self._record_high_risk_prompts(state, kind, images, verdicts)
        raise HighRisk(f"While {'classifying' if who == 'classifier' else 'reviewing the examples'}, the vision "
                       f"model reported that a {kind} image may depict someone under 18. All images from this "
                       "LoRA were deleted.", f"{who} set possible_minor=true")

    def _analyze(self, state: dict, mode: str) -> dict:
        meta, probes = state["meta"], state["stages"]["probe"]["images"]
        ex = self.cfg.examples
        intro = prompts.analysis_intro(meta, self.taxonomy.listing(), ex["min_count"], ex["max_count"])

        def describe(i, img):
            tag = "BASELINE (no LoRA)" if img["baseline"] else f"PROBE with LoRA at weight {img['weight']}"
            return f"Image {i}: {tag}. Prompt: {img['positive']}"

        metadata_only = intro.replace("Probe images follow.", "Probe images are unavailable; classify from metadata alone.")
        if not probes:
            log("  no probe images survived screening; classifying from metadata only")
            result = self.llm.chat_json(prompts.ANALYZE_SYSTEM, metadata_only)
            result["analysis_mode"] = "metadata_only"
            if result.get("possible_minor"):
                self._minor_reported(state, probes, "probe", "classifier", mode)
            return result
        try:
            result = self.llm.chat_json(prompts.ANALYZE_SYSTEM,
                                        [{"type": "text", "text": intro}] + self._images_for_llm(probes, describe))
            result["analysis_mode"] = "images"
        except RefusalError as e:
            log(f"  model declined the images ({e}); falling back to metadata-only analysis")
            result = self.llm.chat_json(prompts.ANALYZE_SYSTEM, intro.replace("Probe images follow.",
                                        "Probe images are unavailable; classify from metadata alone."))
            result["analysis_mode"] = "metadata_only"
        if result.get("possible_minor"):
            self._minor_reported(state, probes, "probe", "classifier", mode)
        return result

    def _example_jobs(self, state: dict) -> list[dict]:
        meta, analysis = state["meta"], state["stages"]["analysis"]
        plan = (analysis.get("example_plan") or {}).get("images") or []
        plan = plan[: self.cfg.examples["max_count"]]
        default_w = (analysis.get("recommended_weight") or {}).get("default") or self.cfg.probe["weight"]
        groups: dict = {}
        jobs = []
        for i, item in enumerate(plan):
            body = str(item.get("prompt", ""))
            missing = [t for t in meta["triggers"] if t.lower() not in body.lower()]
            positive = _join_tags(self.cfg.probe["prefix"], ", ".join(missing), body)
            group = item.get("seed_group")
            seed = groups.setdefault(group, self._seed(meta["id"], 100 + i)) if group else self._seed(meta["id"], 100 + i)
            try:
                weight = max(-3.0, min(3.0, float(item.get("lora_weight", default_w))))
            except (TypeError, ValueError):
                weight = default_w
            aspect = str(item.get("aspect") or "")
            if aspect not in self.cfg.comfy["sizes"] and aspect not in civitai.BUCKET_NAMES:
                aspect = "portrait"
            jobs.append({"positive": positive,
                         "negative": _join_tags(self.cfg.probe["negative"], item.get("negative_extra", "")),
                         "seed": seed, "aspect": aspect, "weight": weight, "baseline": False,
                         "purpose": item.get("purpose", "")})
        return jobs

    def _evaluate(self, state: dict, mode: str) -> dict | None:
        examples = state["stages"]["examples"]["images"]
        if not examples:
            return None

        def describe(i, img):
            return f"Example {i} (weight {img['weight']}). Purpose: {img.get('purpose')}. Prompt: {img['positive']}"

        try:
            result = self.llm.chat_json(prompts.EVALUATE_SYSTEM,
                                        [{"type": "text", "text": prompts.evaluation_intro(state["stages"]["analysis"])}]
                                        + self._images_for_llm(examples, describe))
        except RefusalError as e:
            log(f"  model declined to evaluate examples ({e}); keeping the analysis as final")
            return None
        if result.get("possible_minor"):
            self._minor_reported(state, examples, "example", "evaluator", mode)
        return result

    # ---------- outcomes ----------
    def _discard_images(self, state: dict):
        img_dir = self.cfg.examples_dir / state["meta"]["id"]
        if img_dir.exists():
            shutil.rmtree(self.cfg.assert_writable(img_dir))
        for stage in state["stages"].values():
            if isinstance(stage, dict):
                stage.pop("images", None)
        state["stages"].pop("examples", None)
        state["stages"].pop("evaluation", None)

    def mark_high_risk(self, state: dict, summary: str, detail: str):
        """Set aside as caution, awaiting manual review. All images of this LoRA are deleted."""
        self._discard_images(state)
        state["safety"].update(tier=safety.CAUTION, high_risk_summary=summary, high_risk_detail=detail)
        state["status"] = "caution"
        log(f"  CAUTION: {summary}")

    def mark_quarantined(self, state: dict, summary: str, detail: str):
        self._discard_images(state)
        state["safety"].update(tier=safety.QUARANTINED, high_risk_summary=summary, high_risk_detail=detail)
        state["status"] = "quarantined"
        log(f"  QUARANTINED: {summary}")

    # ---------- driver ----------
    def process(self, meta: dict, previous: dict | None = None) -> dict:
        state = {"meta": meta, "status": "in_progress", "stages": {},
                 "safety": {"screenings": []}, "placement": (previous or {}).get("placement"),
                 "render_settings": {**self.comfy.settings(), "lora": meta["comfy_name"]},
                 "cost_usd": 0.0}
        cost_before = self.llm.total_cost
        log(f"{meta['source_rel']}")
        if (previous or {}).get("status") not in ("done", "needs_review"):
            # Left over from an interrupted attempt (crash or stop): start this LoRA clean.
            stale = self.cfg.examples_dir / meta["id"]
            if stale.exists():
                shutil.rmtree(self.cfg.assert_writable(stale))
                if previous:
                    log("  resuming an interrupted LoRA: cleared its partial images")
        ctx = None
        try:
            try:
                metadata.enrich(meta, self.cfg)
                civ = meta.get("civitai")
                if self.cfg.civitai.get("enabled"):
                    log(f"  civitai: {civ['model_name']} ({len(civ['images'])} example sizes)" if civ
                        else "  civitai: not found")
            except (requests.RequestException, ValueError) as e:
                meta["civitai"] = None
                log(f"  civitai lookup failed ({e}); continuing without it")

            ctx = safety_context(self.cfg, meta, previous)
            if ctx["blocked"]:
                raise Blocked()
            assessment, approval, mode = ctx["assessment"], ctx["approval"], ctx["mode"]
            state["safety"]["assessment"] = assessment
            state["safety"]["metadata_reasons"] = assessment["reasons"]
            state["review"] = approval
            state["render_settings"]["safety_mode"] = mode
            if ctx["quarantine"]:
                q = ctx["quarantine"]
                raise Quarantine(f"Quarantined by manual review on {q['quarantined_at']}: {q.get('notes') or '(no notes)'}",
                                 "manual review decision")
            if assessment["held"] and not approval:
                raise HighRisk(assessment["summary"], "held: " + "; ".join(assessment["reasons"]))
            if approval:
                log(f"  manually reviewed on {approval['approved_at']}: rendering in {mode} mode")
            elif assessment["tier"]:
                log(f"  flagged {assessment['tier']}: rendering in strict mode ({'; '.join(assessment['reasons'])[:200]})")
            if meta["comfy_name"] not in self.comfy.lora_names:
                raise ComfyError(f"ComfyUI doesn't list {meta['comfy_name']!r}; refresh ComfyUI's model list")

            approved = bool(approval)
            state["stages"]["probe"] = {"images": self._render_screen_save(state, self._probe_jobs(meta), "probe",
                                                                           mode, approved)}
            save_state(self.cfg, state)

            log("  classifying")
            state["stages"]["analysis"] = self._analyze(state, mode)
            save_state(self.cfg, state)

            if self.make_examples:
                jobs = self._example_jobs(state)
                rationale = (state["stages"]["analysis"].get("example_plan") or {}).get("rationale")
                state["stages"]["examples"] = {"rationale": rationale,
                                               "images": self._render_screen_save(state, jobs, "example", mode,
                                                                                  approved)}
                save_state(self.cfg, state)
                log("  evaluating examples")
                state["stages"]["evaluation"] = self._evaluate(state, mode)
            state["status"] = "done"
            if state["stages"]["analysis"].get("analysis_mode") == "metadata_only":
                state["status"] = "needs_review"
        except Blocked:
            return reject_blocked(self.cfg, meta, ctx["sha"])
        except Quarantine as e:
            self.mark_quarantined(state, e.summary, e.detail)
        except HighRisk as e:
            self.mark_high_risk(state, e.summary, e.detail)
        except (ComfyError, LLMError, OSError, KeyError, ValueError) as e:
            state["status"] = "error"
            state["error"] = f"{type(e).__name__}: {e}"
            log(f"  ERROR: {state['error']}")
        state["cost_usd"] = round(self.llm.total_cost - cost_before, 5)

        if state["status"] != "error":
            try:
                organize.place(self.cfg, state, self.taxonomy, update_index=False)
            except Exception as e:  # noqa: BLE001 - one bad record must not end a long run
                state["status"] = "error"
                state["error"] = f"could not place output: {type(e).__name__}: {e}"
                log(f"  ERROR: {state['error']}")
        save_state(self.cfg, state)
        organize.write_index(self.cfg)
        return state


def failure_to_high_risk(result: dict, kind: str, total: int, mode: str) -> HighRisk:
    """Turn a failed screening into a plain-language HighRisk explanation."""
    if result.get("error"):
        return HighRisk(f"The {kind} images could not be verified by the age screening (it failed), so they were "
                        "deleted and the LoRA was set aside for manual review.",
                        f"screening error: {result['error']}")
    flagged = result["flagged"]
    why = "; ".join(sorted({str(v.get("reason")) for v in flagged if v.get("reason")})) or "no reason given"
    detail = "; ".join(f"{kind} #{v['index'] + 1}: {v.get('verdict')}/{v.get('sexual')} - {v.get('reason')}"
                       for v in flagged)
    if result["csam_hits"]:
        what = f"as childlike AND {'/'.join(sorted({v['sexual'] for v in result['csam_hits']}))}"
    else:
        what = "as possibly depicting someone under 18"
    return HighRisk(f"The age screener flagged {len(flagged)} of {total} generated {kind} images {what} ({why}). "
                    "All images from this LoRA were deleted.", f"screening flagged ({mode}): {detail}")


def reject_blocked(cfg, meta: dict, sha: str) -> dict:
    """A LoRA on the blocklist reached the pipeline: delete it and record the rejection."""
    deleted = blocklist.delete_verified(cfg, Path(meta["source_path"]), sha)
    blocklist._write_log(cfg, "AUTO-REJECTED", sha, meta["source_path"], f"deleted={deleted}  (found during a run)")
    state = load_state(cfg, meta["id"]) or {"meta": meta, "stages": {}, "safety": {}}
    blocklist._clear_outputs(cfg, state, meta["id"])
    blocklist._mark_state_blocked(cfg, state, sha, "auto-rejected: hash is on the blocklist")
    log(f"  BLOCKED: hash is on the blocklist; file deleted: {deleted}")
    return state


def test_render(cfg, comfy, llm, taxonomy, meta: dict, positive: str, negative_extra: str = "",
                weight: float = 0.9, seed: int | None = None, aspect: str = "1024x1024", count: int = 1,
                save: bool = False) -> dict:
    """Manual test render for the Review tab. Same protections as the pipeline: the LoRA must not
    be blocked or quarantined, a set-aside or held LoRA must be approved first, and every image is
    screened in memory. For an approved LoRA flagged images are hidden and discarded and the rest
    are shown; the approval is kept. For an unapproved LoRA any flag discards the whole batch and
    sets the LoRA aside."""
    metadata.enrich(meta, cfg)
    state = load_state(cfg, meta["id"])
    ctx = safety_context(cfg, meta, state)
    if ctx["blocked"]:
        raise PermissionError("this LoRA is on the blocklist")
    if ctx["quarantine"] or (state or {}).get("status") == "quarantined":
        raise PermissionError("quarantined LoRAs can't be rendered; release it from quarantine first")
    needs_review = ctx["assessment"]["held"] or (state or {}).get("status") == "caution"
    if needs_review and not ctx["approval"]:
        raise PermissionError("this LoRA is set aside (caution); approve it in manual review before testing it")
    mode = ctx["mode"]
    count = max(1, min(4, int(count)))
    seed = int(seed) if seed not in (None, "") else int(time.time() * 1000) % 2**32
    jobs = [{"positive": _join_tags(cfg.probe["prefix"], positive),
             "negative": _join_tags(cfg.probe["negative"], negative_extra),
             "seed": seed + i, "aspect": aspect, "weight": float(weight), "baseline": False} for i in range(count)]
    approved = bool(ctx["approval"])
    proc = Processor(cfg, comfy, llm, taxonomy)
    rendered, result = proc.render_and_screen(meta, jobs, mode, "test", keep_passing=approved)
    verdicts = [result["verdicts"].get(i) for i in range(count)]
    reviews.log_event(cfg, ctx["sha"], meta, "test render", mode=mode, passed=result["passed"],
                      kept=sum(1 for r in rendered if r is not None) if approved else (count if result["passed"] else 0),
                      jobs=[{k: j[k] for k in ("positive", "negative", "seed", "weight", "width", "height")}
                            for j in jobs],
                      verdicts=[{k: (v or {}).get(k) for k in ("verdict", "sexual", "reason")} for v in verdicts])
    out = {"passed": result["passed"], "mode": mode, "error": result.get("error"), "results": [],
           "approved": approved}
    saved_dir = cfg.assert_writable(cfg.tests_dir / meta["id"])
    for i, job in enumerate(jobs):
        v = verdicts[i] or {}
        png = (rendered[i] if approved else (rendered[i] if result["passed"] else None))
        path = None
        if png and save:
            saved_dir.mkdir(parents=True, exist_ok=True)
            path = saved_dir / f"test_{time.strftime('%Y%m%d_%H%M%S')}_{i + 1}.png"
            path.write_bytes(png)
        out["results"].append({"png": png, "path": str(path) if path else None, "seed": job["seed"],
                               "positive": job["positive"], "negative": job["negative"],
                               "verdict": v.get("verdict", NOT_SCREENED), "sexual": v.get("sexual"),
                               "reason": v.get("reason"), "passed": bool(v.get("passed"))})
    if not result["passed"] and not approved:
        err = failure_to_high_risk(result, "test", count, mode)
        state = state or {"meta": meta, "stages": {}, "safety": {"screenings": []}, "placement": None}
        state.setdefault("safety", {}).setdefault("screenings", []).extend(result["screenings"])
        Processor._record_high_risk_prompts(state, "test", jobs, result["verdicts"])
        proc.mark_high_risk(state, err.summary, err.detail)
        organize.place(cfg, state, taxonomy, update_index=False)
        save_state(cfg, state)
        organize.write_index(cfg)
        out["high_risk_summary"] = err.summary
    return out

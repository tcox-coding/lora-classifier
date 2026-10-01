"""The tool's operations, shared by the command line (__main__) and the UI (gui).
All output goes through pipeline.log so either front end can display it."""
import threading
from pathlib import Path

from . import blocklist, metadata, organize, pipeline, reviews, safety
from .comfy import Comfy
from .llm import LLM
from .pipeline import Processor, Stopped, load_state, log, safety_context, save_state
from .taxonomy import Taxonomy

DONE = ("done", "needs_review", "caution", "quarantined", "blocked")  # "pending"/"error" get (re)processed


def select(cfg, match: str | None = None, ids: list[str] | None = None) -> list:
    files = metadata.scan(cfg)
    if match:
        needle = match.replace("\\", "/").lower()
        files = [f for f in files if needle in f.relative_to(cfg.source_dir).as_posix().lower()]
    if ids:
        wanted = set(ids)
        files = [f for f in files if metadata.lora_id(f.relative_to(cfg.source_dir).as_posix()) in wanted]
    return files


def progress(cfg, match: str | None = None) -> dict:
    """How far the library has got: finished (done/needs review), set aside (caution, quarantined,
    blocked), interrupted (a run stopped or crashed mid-LoRA), and never started. A normal run
    processes exactly the interrupted + never-started + errored ones, so it continues where the
    last run left off."""
    states = {s["meta"]["id"]: s for s in organize.all_states(cfg)}
    counts = {"total": 0, "finished": 0, "set_aside": 0, "interrupted": 0, "queued": 0, "errors": 0,
              "not_started": 0}
    for f in select(cfg, match):
        counts["total"] += 1
        s = states.get(metadata.lora_id(f.relative_to(cfg.source_dir).as_posix()))
        status = s["status"] if s else None
        if status in ("done", "needs_review"):
            counts["finished"] += 1
        elif status in ("caution", "quarantined", "blocked"):
            counts["set_aside"] += 1
        elif status == "error":
            counts["errors"] += 1
        elif status == "in_progress":
            counts["interrupted"] += 1
        elif status == "pending":  # sent back to Run (re-tier, approval, release, re-queue)
            counts["queued"] += 1
        else:
            counts["not_started"] += 1
    counts["remaining"] = counts["interrupted"] + counts["queued"] + counts["errors"] + counts["not_started"]
    counts["processed"] = counts["total"] - counts["remaining"]
    return counts


def enforce_blocklist(cfg, dry_run: bool = False) -> list[dict]:
    found = blocklist.enforce(cfg, dry_run=dry_run)
    if found:
        verb = "would be deleted" if dry_run else "deleted"
        log(f"blocklist: {len(found)} blocked LoRA file(s) found and {verb}")
    return found


def run(cfg, match=None, limit=None, probes=None, redo=False, no_examples=False,
        should_stop=None, on_progress=None, ids=None) -> dict:
    """Probe, classify, describe and illustrate LoRAs. Resumable; returns status counts.
    on_progress(done, total, current_name) is called before each LoRA and at the end."""
    should_stop = should_stop or (lambda: False)
    enforce_blocklist(cfg)  # blocked hashes are removed before anything else happens
    comfy = Comfy(cfg)
    comfy.check()
    llm = LLM(cfg)
    taxonomy = Taxonomy(cfg)
    proc = Processor(cfg, comfy, llm, taxonomy, probe_count=probes, make_examples=not no_examples,
                     should_stop=should_stop)

    todo = []
    for f in select(cfg, match, ids):
        meta = metadata.extract(f, cfg)
        prev = load_state(cfg, meta["id"])
        if prev and prev["status"] in DONE and not redo:
            continue
        todo.append((meta, prev))
    if not redo:
        p = progress(cfg, match)
        log(f"{p['processed']} of {p['total']} LoRA(s) already processed and skipped; continuing with "
            f"{p['remaining']} ({p['not_started']} not started, {p['queued']} queued, {p['interrupted']} interrupted, "
            f"{p['errors']} errors)")
    if limit:
        todo = todo[:limit]
    log(f"{len(todo)} LoRA(s) to process")
    counts: dict[str, int] = {}
    try:
        for n, (meta, prev) in enumerate(todo):
            if should_stop():
                raise Stopped()
            if on_progress:
                on_progress(n, len(todo), meta["source_rel"])
            log(f"--- {n + 1}/{len(todo)}")
            state = proc.process(meta, prev)
            counts[state["status"]] = counts.get(state["status"], 0) + 1
    except Stopped:
        log("stopped; the unfinished LoRA will resume on the next run")
        counts["stopped"] = 1
    if on_progress:
        on_progress(sum(v for k, v in counts.items() if k != "stopped"), len(todo), "")
    log(f"finished: {counts}  |  DeepInfra cost this run: ${llm.total_cost:.4f}")
    log(f"catalogue: {cfg.catalog_dir / 'index.md'}")
    return counts


def scan(cfg, match=None) -> int:
    """Metadata preview: no rendering or DeepInfra calls, nothing deleted. Looks LoRAs up on Civitai
    (if enabled; the first time hashes each file). Returns the number that would be set aside."""
    set_aside = 0
    files = select(cfg, match)
    for f in files:
        meta = metadata.extract(f, cfg)
        try:
            metadata.enrich(meta, cfg)  # first time: hashes the file and asks Civitai (cached after)
        except Exception as e:  # noqa: BLE001 - preview keeps going without Civitai
            log(f"  civitai lookup failed for {meta['source_rel']}: {e}")
        ctx = safety_context(cfg, meta, load_state(cfg, meta["id"]))
        tier = ctx["assessment"]["tier"]
        set_aside += bool(tier) or ctx["blocked"]
        flag = ""
        if ctx["blocked"]:
            flag = "  BLOCKED: hash is on the blocklist (deleted on the next run)"
        elif ctx["quarantine"]:
            flag = "  QUARANTINED (manual review)"
        elif tier:
            flag = f"  {'HELD ' if ctx['assessment']['held'] else ''}{tier.upper()}" \
                   f"{' (approved)' if ctx['approval'] else ''}: {ctx['assessment']['summary']}"
        civ = meta.get("civitai")
        sizes = sorted({f"{i['bucket'][0]}x{i['bucket'][1]}" for i in civ["images"]}) if civ else "not on Civitai"
        log(f"{meta['source_rel']}\n    triggers={meta['triggers']} top={meta['top_tags'][:5]} civitai={sizes}{flag}")
    log(f"{len(files)} LoRA(s) scanned, {set_aside} flagged, quarantined or blocked")
    return set_aside


def status(cfg, verbose=False) -> dict:
    states = organize.all_states(cfg)
    total = len(metadata.scan(cfg))
    counts: dict[str, int] = {}
    for s in states:
        counts[s["status"]] = counts.get(s["status"], 0) + 1
    log(f"{len(states)}/{total} processed: {counts}")
    log(f"total DeepInfra cost: ${sum(s.get('cost_usd') or 0 for s in states):.4f}")
    quiet = ("caution", "quarantined", "blocked")
    for s in states:
        if s["status"] in ("error", "needs_review") + quiet and (verbose or s["status"] not in quiet):
            reason = s.get("error") or s["safety"].get("high_risk_summary") or ""
            log(f"  [{s['status']}] {s['meta']['source_rel']}: {reason}")
    return counts


def rebuild(cfg):
    organize.rebuild(cfg, Taxonomy(cfg))
    log(f"rebuilt {cfg.output_dir / 'loras'} and {cfg.descriptions_dir}")


# ---------------------------------------------------------------- manual review

def current_meta(cfg, lora_id: str) -> dict:
    """Fresh metadata for a LoRA (from its file if it still exists, else from its state)."""
    state = load_state(cfg, lora_id)
    if state:
        source = Path(state["meta"]["source_path"])
        if not source.exists():
            return state["meta"]
    else:  # never processed: find the file in the library
        found = select(cfg, ids=[lora_id])
        if not found:
            raise KeyError(f"no LoRA with id {lora_id}")
        source = found[0]
    stat = source.stat()
    key = (str(source), stat.st_size, stat.st_mtime_ns)
    with _meta_lock:
        cached = _meta_cache.get(key)
    if cached is None:
        cached = metadata.extract(source, cfg)
        try:
            metadata.enrich(cached, cfg)
        except Exception as e:  # noqa: BLE001 - review works without Civitai
            log(f"civitai lookup failed: {e}")
        with _meta_lock:
            _meta_cache[key] = cached
    return dict(cached)  # callers only set top-level fields (e.g. sha256) on their copy


# Parsed metadata per (path, size, mtime): reading every LoRA header and scanning all its caption
# tags takes ~20 ms each, so re-deriving it on every UI refresh froze the window for 15 s.
_meta_cache: dict = {}
_meta_lock = threading.Lock()


SET_ASIDE_DETAILS = ("metadata gate", "retroactive metadata gate", "held:")


def review_items(cfg) -> list[dict]:
    """Everything the Review tab lists: flagged (reviewable/caution), set aside, approved, quarantined
    or blocked LoRAs. Flags are computed fresh from the file, so older records follow today's rules."""
    items = []
    for s in organize.all_states(cfg):
        meta = s["meta"]
        item = {"id": meta["id"], "name": meta["filename"], "rel": meta["source_rel"], "status": s["status"],
                "sha256": meta.get("sha256") or (s.get("safety", {}).get("blocked") or {}).get("sha256"),
                "summary": s.get("safety", {}).get("high_risk_summary"), "tier": None, "held": False,
                "assessment": None, "quarantine": None}
        if s["status"] != "blocked" and Path(meta["source_path"]).exists():
            fresh = current_meta(cfg, meta["id"])
            ctx = safety_context(cfg, fresh, s)
            a = ctx["assessment"]
            item.update(sha256=ctx["sha"], tier=a["tier"], held=a["held"], assessment=a, quarantine=ctx["quarantine"])
            if s["status"] in ("done", "needs_review", "pending", "error") and a["tier"]:
                item["summary"] = a["summary"]
            detail = s.get("safety", {}).get("high_risk_detail") or ""
            if s["status"] == "caution" and detail.startswith(SET_ASIDE_DETAILS) and not a["held"]:
                item["summary"] = (a["summary"] if a["tier"] else "No longer flagged.") + \
                    " (Set aside under older rules without rendering; a re-tier queues it for processing.)"
        item["approval"] = reviews.approval(cfg, item["sha256"])
        if item["quarantine"] is None:
            item["quarantine"] = reviews.quarantine_record(cfg, item["sha256"])
        if s["status"] in ("caution", "quarantined", "blocked") or item["approval"] or item["tier"] or item["quarantine"]:
            items.append(item)
    return items


def approve(cfg, lora_id: str, notes: str):
    meta = current_meta(cfg, lora_id)
    ctx = safety_context(cfg, meta, load_state(cfg, lora_id))
    if ctx["blocked"]:
        raise PermissionError("this LoRA is on the blocklist")
    reviews.approve(cfg, ctx["sha"], meta, notes, ctx["assessment"])
    mode = safety.render_mode(ctx["assessment"], True)
    state = load_state(cfg, lora_id)
    if state and state["status"] == "caution":
        _queue(cfg, state, meta)  # back to the normal flow: Continue / Reprocess picks it up
        log(f"approved: {meta['source_rel']} (renders in {mode} mode); queued for processing, "
            "press Continue or Reprocess")
    else:
        log(f"approved: {meta['source_rel']} (renders in {mode} mode)")


def revoke(cfg, lora_id: str, reason: str = "revoked manually"):
    meta = current_meta(cfg, lora_id)
    sha = meta.get("sha256") or safety_context(cfg, meta)["sha"]
    if reviews.revoke(cfg, sha, meta, reason):
        log(f"approval revoked: {meta['source_rel']}")


def quarantine(cfg, lora_id: str, notes: str):
    """Manual review decision: quarantine a LoRA. Its images are deleted and it is never rendered."""
    meta = current_meta(cfg, lora_id)
    state = load_state(cfg, lora_id) or {"meta": meta, "stages": {}, "safety": {}, "placement": None}
    ctx = safety_context(cfg, meta, state)
    reviews.quarantine(cfg, ctx["sha"], meta, notes)
    state["meta"] = meta
    state.setdefault("safety", {})["assessment"] = ctx["assessment"]
    taxonomy = Taxonomy(cfg)
    Processor(cfg, None, None, taxonomy).mark_quarantined(
        state, f"Quarantined by manual review on {reviews.now()}: {notes or '(no notes)'}", "manual review decision")
    organize.place(cfg, state, taxonomy, update_index=False)
    save_state(cfg, state)
    organize.write_index(cfg)


def release(cfg, lora_id: str, notes: str):
    """Undo a quarantine. The LoRA is queued ("pending") and processed again on the next run."""
    meta = current_meta(cfg, lora_id)
    sha = safety_context(cfg, meta)["sha"]
    reviews.release(cfg, sha, meta, notes)
    state = load_state(cfg, lora_id)
    if state:
        _queue(cfg, state, meta)
    log(f"released from quarantine: {meta['source_rel']} (it will be processed on the next run)")


def _queue(cfg, state: dict, meta: dict):
    """Remove a LoRA's placement and mark it pending so the next run processes it again."""
    taxonomy = Taxonomy(cfg)
    for key in ("lora", "description_json", "description_md"):
        organize._unlink(cfg, (state.get("placement") or {}).get(key))
    if (state.get("placement") or {}).get("counted"):
        taxonomy.remove(state["placement"]["category_path"])
    state.update(meta=meta, status="pending", placement=None, stages={})
    save_state(cfg, state)
    organize.write_index(cfg)


def block(cfg, lora_id: str, reason: str) -> dict:
    """Blocklist a LoRA and permanently delete its .safetensors file."""
    meta = current_meta(cfg, lora_id)
    return blocklist.block(cfg, meta, reason, Taxonomy(cfg))


def test(cfg, lora_id: str, **params) -> dict:
    comfy = Comfy(cfg)
    comfy.check()
    meta = current_meta(cfg, lora_id)
    return pipeline.test_render(cfg, comfy, LLM(cfg), Taxonomy(cfg), meta, **params)


def reprocess(cfg, lora_id: str, **kwargs) -> dict:
    """Run one LoRA through the full pipeline again (e.g. after approving it)."""
    return run(cfg, ids=[lora_id], redo=True, **kwargs)


def retier(cfg, dry_run: bool = True) -> list[dict]:
    """Bring records made under older rules in line with the current ones. Dry run by default.
      - set aside only because of metadata (never rendered) and no longer held -> pending (re-run)
      - finished but now held (explicit under-18 ages) and not approved      -> caution
      - finished and flagged differently than recorded                      -> description refreshed
    Screener-flagged LoRAs stay set aside: their images really did look childlike."""
    changes = []
    taxonomy = None if dry_run else Taxonomy(cfg)
    for s in organize.all_states(cfg):
        if s["status"] not in ("done", "needs_review", "caution"):
            continue
        if not Path(s["meta"]["source_path"]).exists():
            continue
        meta = current_meta(cfg, s["meta"]["id"])
        ctx = safety_context(cfg, meta, s)
        a = ctx["assessment"]
        if ctx["quarantine"]:
            continue
        detail = s.get("safety", {}).get("high_risk_detail") or ""
        old_tier = (s.get("safety", {}).get("assessment") or {}).get("tier")
        new = None
        if s["status"] == "caution" and detail.startswith(SET_ASIDE_DETAILS) and not (a["held"] and not ctx["approval"]):
            new = "pending"
        elif s["status"] in ("done", "needs_review") and a["held"] and not ctx["approval"]:
            new = "caution"
        elif s["status"] in ("done", "needs_review") and a["tier"] != old_tier:
            new = f"{s['status']} (flag: {a['tier'] or 'none'})"
        if not new:
            continue
        changes.append({"id": meta["id"], "file": meta["source_rel"], "from": s["status"], "to": new,
                        "summary": a["summary"]})
        if dry_run:
            continue
        if new == "pending":
            _queue(cfg, s, meta)
            continue
        s["meta"] = meta
        s["safety"]["assessment"] = a
        s["safety"]["metadata_reasons"] = a["reasons"]
        if new == "caution":
            Processor(cfg, None, None, taxonomy).mark_high_risk(s, a["summary"], "held: " + "; ".join(a["reasons"]))
        organize.place(cfg, s, taxonomy, update_index=False)
        save_state(cfg, s)
    if not dry_run:
        organize.write_index(cfg)
    for c in changes:
        log(f"{'would change' if dry_run else 'changed'} {c['file']}: {c['from']} -> {c['to']}")
    log(f"{len(changes)} LoRA(s) {'would change' if dry_run else 'changed'}")
    return changes

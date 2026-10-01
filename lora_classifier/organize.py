"""Places LoRAs into output/loras/<category>/ and writes descriptions + index. Never touches the source."""
import json
import os
import shutil
from pathlib import Path

from .config import CAUTION_FOLDER, QUARANTINE_FOLDER
from . import store
from .taxonomy import clean_path

PLACED_STATUSES = ("done", "needs_review", "caution", "quarantined")
SET_ASIDE = {"caution": CAUTION_FOLDER, "quarantined": QUARANTINE_FOLDER}


def final_record(state: dict) -> dict:
    """Merge analysis + evaluation into the catalogue entry."""
    record = dict(state["stages"].get("analysis") or {})
    evaluation = state["stages"].get("evaluation") or {}
    record.pop("example_plan", None)
    for key, value in evaluation.items():
        if key not in ("per_example", "possible_minor") and value not in (None, "", [], {}):
            record[key] = value
    record["category_path"] = clean_path(record.get("category_path"))
    return record


def _unlink(cfg, path: str | None):
    if path and Path(path).exists():
        target = cfg.assert_writable(path)
        target.unlink()
        # Remove folders the move left empty (never above output/loras or the descriptions root).
        stop = {cfg.loras_out.resolve(), cfg.descriptions_dir.resolve()}
        parent = target.parent
        while parent not in stop and cfg.output_dir in parent.parents and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent


def _place_file(cfg, src: Path, category: list[str], lora_id: str) -> Path | None:
    if cfg.link_mode == "none":
        return None
    folder = cfg.loras_out.joinpath(*category)
    dst = folder / src.name
    if dst.exists():  # a different LoRA with the same filename already lives here
        dst = folder / f"{src.stem}__{lora_id[-8:]}{src.suffix}"
    if len(str(dst)) > 250:
        dst = folder / f"{src.stem[:40]}__{lora_id[-8:]}{src.suffix}"
    dst = cfg.assert_writable(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if cfg.link_mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    elif cfg.link_mode == "symlink":
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)
    return dst


def _image_entries(images: list[dict], per_example: list[dict], md_dir: Path) -> list[dict]:
    notes = {e.get("index"): e for e in per_example or [] if isinstance(e, dict)}
    out = []
    for i, img in enumerate(images or []):
        entry = {
            "path": img["path"],
            "relative_path": os.path.relpath(img["path"], md_dir).replace("\\", "/"),
            "with_lora": not img["baseline"],
            "purpose": img.get("purpose"),
            "prompt": img["positive"],
            "negative": img["negative"],
            "seed": img["seed"], "lora_weight": 0.0 if img["baseline"] else img["weight"], "aspect": img["aspect"],
            "width": img.get("width"), "height": img.get("height"),
        }
        if i in notes:
            entry.update({k: notes[i].get(k) for k in ("success", "quality", "notes")})
        out.append(entry)
    return out


def _civitai_summary(civ: dict | None) -> dict | None:
    if not civ:
        return None
    sizes: dict[str, int] = {}
    for img in civ.get("images", []):
        key = f"{img['bucket'][0]}x{img['bucket'][1]}"
        sizes[key] = sizes.get(key, 0) + 1
    url = None
    if civ.get("model_id"):
        url = f"https://civitai.com/models/{civ['model_id']}"
        if civ.get("version_id"):
            url += f"?modelVersionId={civ['version_id']}"
    return {"model": civ.get("model_name"), "version": civ.get("version_name"), "url": url,
            "trained_words": civ.get("trained_words"), "example_resolutions": sizes,
            "examples_marked_minor": sum(1 for img in civ.get("images", []) if img.get("minor"))}


def build_description(state: dict, lora_file: Path | None, md_dir: Path) -> dict:
    meta = state["meta"]
    base = {
        "name": Path(meta["filename"]).stem,
        "id": meta["id"],
        "status": state["status"],
        "lora_file": str(lora_file) if lora_file else None,
        "source_file": meta["source_path"],
        "previous_folder": meta["source_folder"],
        "trigger_words_detected": meta["triggers"],
        "training": {"images": meta.get("training_images"), "top_tags": meta.get("top_tags"),
                     "base_model": meta.get("base_model"), "network_dim": meta.get("network_dim")},
        "cost_usd": state.get("cost_usd"),
        "render_settings": state.get("render_settings"),
        "civitai": _civitai_summary(meta.get("civitai")),
        "sha256": meta.get("sha256"),
        "review": state.get("review"),
        "safety_mode": (state.get("render_settings") or {}).get("safety_mode"),
        "assessment": (state.get("safety") or {}).get("assessment"),
        # Images the screener discarded for an approved LoRA (never saved); prompts kept for reference.
        "discarded_images": (state.get("safety") or {}).get("discarded_prompts", []),
    }
    if state["status"] in SET_ASIDE:
        base.update({
            "category": SET_ASIDE[state["status"]],
            "high_risk_summary": state["safety"].get("high_risk_summary"),
            "high_risk_detail": state["safety"].get("high_risk_detail"),
            "metadata_reasons": state["safety"].get("metadata_reasons"),
            # Prompts that were rendered with this LoRA. Their images were deleted.
            "high_risk_prompts": state["safety"].get("high_risk_prompts", []),
        })
        return base

    record = final_record(state)
    evaluation = state["stages"].get("evaluation") or {}
    examples = (state["stages"].get("examples") or {}).get("images")
    base.update({
        "category": "/".join(record["category_path"]),
        "folder_changed": "/".join(record["category_path"]) != "/".join(clean_path(meta["source_folder"])),
        **{k: record.get(k) for k in (
            "category_path", "category_reason", "confidence", "lora_type", "summary", "description",
            "trigger_words", "recommended_weight", "nsfw_level", "metrics", "extra_metrics", "scores",
            "strengths", "weaknesses", "usage_tips", "analysis_mode")},
        "example_rationale": (state["stages"].get("examples") or {}).get("rationale"),
        "example_images": _image_entries(examples, evaluation.get("per_example"), md_dir),
        "probe_images": _image_entries(state["stages"]["probe"]["images"], None, md_dir),
        "safety": {"screened_images": sum(len(s["verdicts"]) for s in state["safety"]["screenings"]),
                   "all_screened_adult": True},
    })
    return base


def _md_list(items) -> str:
    return "\n".join(f"- {x}" for x in items or []) or "- (none)"


def _md_value(v) -> str:
    if isinstance(v, dict):
        return "; ".join(f"{k}: {_md_value(x)}" for k, x in v.items() if x not in (None, "", []))
    if isinstance(v, list):
        return ", ".join(str(x) for x in v)
    return str(v)


def _settings_line(d: dict) -> str:
    rs = d.get("render_settings")
    if not rs:
        return "_Generation settings not recorded._"
    note = f" ({rs['note']})" if rs.get("note") else ""
    return ("**Generation settings:** " + ", ".join(f"{k} `{v}`" for k, v in rs.items() if k != "note") + note)


def _size(p: dict) -> str:
    return f"{p['width']}x{p['height']}" if p.get("width") else p.get("aspect", "")


def render_markdown(d: dict) -> str:
    lines = [f"# {d['name']}", ""]
    lines += [f"- **Status:** {d['status']}", f"- **Category:** `{d['category']}`",
              f"- **Previously in:** `{d['previous_folder']}`",
              f"- **LoRA file:** `{d['lora_file']}`", f"- **Source:** `{d['source_file']}`"]
    civ = d.get("civitai")
    if civ and civ.get("url"):
        lines.append(f"- **Civitai:** [{civ['model']} ({civ['version']})]({civ['url']})")
    else:
        lines.append(f"- **Civitai:** {civ['model'] if civ else 'not found'}")
    rv = d.get("review")
    lines += [f"- **Manual review:** approved {rv['approved_at']}: {rv.get('notes') or '(no notes)'}" if rv
              else "- **Manual review:** none", f"- **Safety mode:** {d.get('safety_mode') or 'standard'}", ""]
    if d["status"] in SET_ASIDE:
        heading = ("QUARANTINED by manual review" if d["status"] == "quarantined"
                   else "CAUTION: why this LoRA was set aside")
        lines += [f"## {heading}", "", d.get("high_risk_summary") or "", "",
                  f"_Technical detail: {d.get('high_risk_detail')}_", ""]
        if d.get("metadata_reasons"):
            lines += ["**Metadata reasons:**", _md_list(d["metadata_reasons"]), ""]
        prompts = d.get("high_risk_prompts") or []
        lines += ["## Prompts rendered with this LoRA", "",
                  "All images were deleted. These are the prompts that produced them." if prompts
                  else "No images were rendered (blocked before generation).", ""]
        if prompts:
            lines += [_settings_line(d), ""]
        for p in prompts:
            if p["flagged"]:
                flag = f"**FLAGGED**: {p['verdict']}"
            elif p["verdict"] == "not screened":
                flag = "not screened, discarded with its batch"
            else:
                flag = f"passed: {p['verdict']}"
            lines += [f"### {p['kind']} #{p['number']} ({flag}{' - ' + p['reason'] if p.get('reason') else ''})",
                      f"- seed {p['seed']}, weight {p['weight']}, {_size(p)}",
                      f"- positive: `{p['positive']}`", f"- negative: `{p['negative']}`", ""]
        return "\n".join(lines)

    rw = d.get("recommended_weight") or {}
    flag = (d.get("assessment") or {})
    if flag.get("tier"):
        lines += [f"> **Safety flag ({flag['tier']}):** {flag.get('summary')}", ""]
    lines += [f"> {d.get('summary') or ''}", "", d.get("description") or "", "",
              f"- **Type:** {d.get('lora_type')}  |  **NSFW:** {d.get('nsfw_level')}  |  "
              f"**Confidence:** {d.get('confidence')}  |  **Analysis:** {d.get('analysis_mode')}",
              f"- **Trigger words:** {_md_value(d.get('trigger_words') or d['trigger_words_detected'])}",
              f"- **Weight:** min {rw.get('min')} / default {rw.get('default')} / max {rw.get('max')}",
              f"- **Why this category:** {d.get('category_reason')}", "", "## Metrics", "",
              "| metric | value |", "|---|---|"]
    for name, value in (d.get("metrics") or {}).items():
        lines.append(f"| {name} | {_md_value(value)} |")
    for name, value in (d.get("extra_metrics") or {}).items():
        lines.append(f"| {name} | {_md_value(value)} |")
    lines += ["", "## Scores", "", "| score | /10 |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in (d.get("scores") or {}).items()]
    lines += ["", "## Strengths", _md_list(d.get("strengths")), "", "## Weaknesses", _md_list(d.get("weaknesses")), ""]
    if d.get("usage_tips"):
        lines += ["## Usage tips", _md_list(d["usage_tips"]), ""]
    lines += ["## Example images", "", d.get("example_rationale") or "", ""]
    for i, img in enumerate(d.get("example_images") or [], 1):
        verdict = f" — quality {img.get('quality')}/10: {img.get('notes')}" if img.get("quality") else ""
        lines += [f"### Example {i}: {img.get('purpose') or ''}{verdict}", "",
                  f"![example {i}]({img['relative_path'].replace(' ', '%20')})", "",
                  f"`{img['prompt']}` (weight {img['lora_weight']}, seed {img['seed']})", ""]
    discarded = d.get("discarded_images") or []
    if discarded:
        lines += [f"## Discarded by the screener ({len(discarded)})", "",
                  "These images were flagged, never saved or shown. The LoRA is approved, so the rest were kept.", ""]
        lines += [f"- {p['kind']} #{p['number']} ({p.get('verdict')}/{p.get('sexual')}): {p.get('reason') or ''} "
                  f"— seed {p['seed']}, `{p['positive']}`" for p in discarded]
        lines.append("")
    lines += [_settings_line(d), "", "## Probe images", ""]
    for i, img in enumerate(d.get("probe_images") or [], 1):
        label = "probe" if img["with_lora"] else "baseline (no LoRA)"
        lines += [f"- {label} {i}: [{Path(img['path']).name}]({img['relative_path'].replace(' ', '%20')}) — `{img['prompt']}`"]
    return "\n".join(lines) + "\n"


def place(cfg, state: dict, taxonomy, update_index: bool = True):
    meta = state["meta"]
    old = state.get("placement") or {}
    for key in ("lora", "description_json", "description_md"):
        _unlink(cfg, old.get(key))
    if old.get("category_path") and old.get("counted"):
        taxonomy.remove(old["category_path"])

    if state["status"] in SET_ASIDE:
        category = [SET_ASIDE[state["status"]]] + (clean_path(meta["source_folder"]) if meta["source_folder"] != "." else [])
        counted = False
    else:
        category = final_record(state)["category_path"]
        taxonomy.add(category)
        counted = True

    lora_file = _place_file(cfg, Path(meta["source_path"]), category, meta["id"])
    desc_dir = cfg.assert_writable(cfg.descriptions_dir.joinpath(*category))
    desc_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(lora_file).stem if lora_file else Path(meta["filename"]).stem
    desc = build_description(state, lora_file, desc_dir)
    json_path, md_path = desc_dir / f"{stem}.json", desc_dir / f"{stem}.md"
    store.write_json_atomic(json_path, desc)
    md_path.write_text(render_markdown(desc), encoding="utf-8")

    state["placement"] = {"category_path": category, "counted": counted, "lora": str(lora_file) if lora_file else None,
                          "description_json": str(json_path), "description_md": str(md_path)}
    if update_index:
        write_index(cfg)


def all_states(cfg) -> list[dict]:
    return store.all_states(cfg)


def write_index(cfg):
    rows = []
    for s in all_states(cfg):
        pl = s.get("placement") or {}
        rows.append({"id": s["meta"]["id"], "name": s["meta"]["filename"], "status": s["status"],
                     "category": "/".join(pl.get("category_path") or []), "previous_folder": s["meta"]["source_folder"],
                     "summary": final_record(s).get("summary") if s["status"] in ("done", "needs_review") else
                     s.get("safety", {}).get("high_risk_summary") or s.get("error"),
                     "description_md": pl.get("description_md")})
    cfg.catalog_dir.mkdir(parents=True, exist_ok=True)
    cfg.assert_writable(cfg.catalog_dir / "index.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = ["# LoRA catalogue", ""]
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    lines += [" | ".join(f"**{k}**: {v}" for k, v in sorted(counts.items())), ""]
    current = None
    for r in sorted((r for r in rows if r["description_md"]), key=lambda r: (r["category"], r["name"].lower())):
        if r["category"] != current:
            current = r["category"]
            lines += ["", f"## {current}", ""]
        link = os.path.relpath(r["description_md"], cfg.catalog_dir).replace("\\", "/").replace(" ", "%20")
        moved = "" if r["category"] == "/".join(clean_path(r["previous_folder"])) else f" _(was `{r['previous_folder']}`)_"
        flag = "" if r["status"] == "done" else f" **[{r['status']}]**"
        lines.append(f"- [{r['name']}]({link}){flag}{moved} — {r['summary'] or ''}")
    blocked = [r for r in rows if r["status"] == "blocked"]
    if blocked:
        lines += ["", "## Blocked (file deleted, hash on the blocklist)", ""]
        lines += [f"- {r['name']}" for r in blocked]
    errors = [r for r in rows if r["status"] == "error"]
    if errors:
        lines += ["", "## Errors (will be retried on next run)", ""]
        lines += [f"- {r['name']}: {r['summary']}" for r in errors]
    (cfg.catalog_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def rebuild(cfg, taxonomy):
    """Regenerate output/loras and descriptions from saved state (e.g. after editing categories)."""
    for d in (cfg.output_dir / "loras", cfg.descriptions_dir):
        if d.exists():
            shutil.rmtree(cfg.assert_writable(d))
    taxonomy.paths = {k: 0 for k in taxonomy.paths}
    from .pipeline import save_state
    for s in all_states(cfg):
        if s["status"] in PLACED_STATUSES:
            s["placement"] = None
            place(cfg, s, taxonomy, update_index=False)
            save_state(cfg, s)
    write_index(cfg)

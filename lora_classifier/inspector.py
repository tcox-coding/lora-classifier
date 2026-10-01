"""Inspector tab: browse classified LoRAs by category, read their descriptions and flip
through their example images. Reads the description JSONs written by organize.py."""
import json
import os
import subprocess
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from PIL import Image, ImageTk

from .config import CAUTION_FOLDER, QUARANTINE_FOLDER
from .organize import _md_value as fmt

STATUS_FILTERS = ["All", "done", "needs_review", "flagged", "caution", "quarantined"]
SET_ASIDE = {"caution": CAUTION_FOLDER, "quarantined": QUARANTINE_FOLDER, "high_risk": CAUTION_FOLDER}


def load_descriptions(cfg) -> list[dict]:
    out = []
    if cfg.descriptions_dir.exists():
        for p in sorted(cfg.descriptions_dir.rglob("*.json")):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            d["_json_path"] = str(p)
            out.append(d)
    return out


def settings_text(d: dict) -> str:
    rs = d.get("render_settings")
    if not rs:
        return "Generation settings not recorded."
    text = "  ·  ".join(f"{k}: {v}" for k, v in rs.items() if k != "note")
    return text + (f"\n({rs['note']})" if rs.get("note") else "")


def tree_path(d: dict) -> list[str]:
    """Where a LoRA sits in the tree. High-risk LoRAs keep their original folder underneath."""
    if d.get("status") in SET_ASIDE:
        prev = d.get("previous_folder") or ""
        return [SET_ASIDE[d["status"]]] + ([p for p in prev.split("/") if p and p != "."])
    return (d.get("category") or "uncategorized").split("/")


def image_choices(d: dict) -> list[tuple[str, dict]]:
    """(label, image entry) pairs for the dropdown: examples first, then probes/baselines."""
    choices = []
    for i, img in enumerate(d.get("example_images") or [], 1):
        quality = f" ({img['quality']}/10)" if img.get("quality") else ""
        choices.append((f"Example {i}: {img.get('purpose') or ''}{quality}", img))
    probe_n = base_n = 0
    for img in d.get("probe_images") or []:
        if img.get("with_lora", True):
            probe_n += 1
            choices.append((f"Probe {probe_n}", img))
        else:
            base_n += 1
            choices.append((f"Baseline {base_n} (no LoRA)", img))
    return choices


class Inspector:
    def __init__(self, parent: ttk.Frame, cfg, on_block=None, on_refresh=None):
        self.cfg = cfg
        self.on_block = on_block
        self.on_refresh = on_refresh  # the app reloads data in the background and calls apply()
        self.records: dict[str, dict] = {}
        self.choices: list[tuple[str, dict]] = []
        self.current: dict | None = None
        self._photo = None
        self._resize_job = None

        # ---- toolbar ----
        bar = ttk.Frame(parent, padding=(0, 0, 0, 8))
        bar.pack(fill="x")
        ttk.Label(bar, text="Search:").pack(side="left")
        self.search = tk.StringVar()
        self.search.trace_add("write", lambda *_: self.populate())
        ttk.Entry(bar, textvariable=self.search, width=30).pack(side="left", padx=(4, 12))
        ttk.Label(bar, text="Status:").pack(side="left")
        self.status_filter = tk.StringVar(value="All")
        status_box = ttk.Combobox(bar, textvariable=self.status_filter, values=STATUS_FILTERS,
                                  width=13, state="readonly")
        status_box.pack(side="left", padx=4)
        status_box.bind("<<ComboboxSelected>>", lambda _e: self.populate())
        ttk.Button(bar, text="Refresh", command=self.refresh).pack(side="left", padx=8)
        self.count_label = ttk.Label(bar, foreground="gray")
        self.count_label.pack(side="right")

        # ---- panes: tree | details + images ----
        panes = ttk.PanedWindow(parent, orient="horizontal")
        panes.pack(fill="both", expand=True)

        tree_frame = ttk.Frame(panes)
        self.tree = ttk.Treeview(tree_frame, show="tree", selectmode="browse")
        tree_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        tree_scroll.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self.on_select)
        self.tree.tag_configure("caution", foreground="#b00020")
        self.tree.tag_configure("high_risk", foreground="#b00020")
        self.tree.tag_configure("flagged", foreground="#9a6700")
        self.tree.tag_configure("quarantined", foreground="#7a0010")
        self.tree.tag_configure("needs_review", foreground="#9a6700")
        self.tree.tag_configure("category", foreground="#555555")
        panes.add(tree_frame, weight=1)

        right = ttk.PanedWindow(panes, orient="vertical")
        panes.add(right, weight=3)

        text_frame = ttk.Frame(right)
        self.text = tk.Text(text_frame, wrap="word", state="disabled", font=("Segoe UI", 10),
                            padx=10, pady=8, relief="flat", height=9)
        text_scroll = ttk.Scrollbar(text_frame, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=text_scroll.set)
        self.text.pack(side="left", fill="both", expand=True)
        text_scroll.pack(side="right", fill="y")
        self.text.tag_configure("title", font=("Segoe UI", 15, "bold"), spacing3=4)
        self.text.tag_configure("h", font=("Segoe UI", 10, "bold"), spacing1=8, spacing3=2)
        self.text.tag_configure("muted", foreground="gray")
        self.text.tag_configure("summary", font=("Segoe UI", 10, "italic"), spacing3=6)
        self.text.tag_configure("warn", foreground="#b00020", font=("Segoe UI", 10, "bold"))
        self.text.tag_configure("mono", font=("Consolas", 9))
        right.add(text_frame, weight=1)

        img_frame = ttk.Frame(right, padding=(0, 6, 0, 0))
        right.add(img_frame, weight=2)
        picker = ttk.Frame(img_frame)
        picker.pack(fill="x")
        ttk.Label(picker, text="Image:").pack(side="left")
        self.image_var = tk.StringVar()
        self.image_box = ttk.Combobox(picker, textvariable=self.image_var, state="readonly")
        self.image_box.pack(side="left", fill="x", expand=True, padx=4)
        self.image_box.bind("<<ComboboxSelected>>", lambda _e: self.show_image())
        ttk.Button(picker, text="◀", width=3, command=lambda: self.step_image(-1)).pack(side="left")
        ttk.Button(picker, text="▶", width=3, command=lambda: self.step_image(1)).pack(side="left", padx=(2, 8))
        ttk.Button(picker, text="Open image", command=self.open_image).pack(side="left")

        self.image_label = ttk.Label(img_frame, anchor="center", text="Select a LoRA", foreground="gray")
        self.image_label.pack(fill="both", expand=True, pady=6)
        self.image_label.bind("<Configure>", self.on_resize)
        self.image_label.bind("<Double-Button-1>", lambda _e: self.open_image())
        self.image_caption = ttk.Label(img_frame, wraplength=600, foreground="gray", justify="left")
        self.image_caption.pack(fill="x")
        img_frame.bind("<Configure>", lambda e: self.image_caption.config(wraplength=max(e.width - 20, 200)))

        files = ttk.Frame(parent, padding=(0, 8, 0, 0))
        files.pack(fill="x")
        ttk.Button(files, text="Open description", command=self.open_description).pack(side="left")
        ttk.Button(files, text="Show LoRA file in Explorer", command=self.show_lora_file).pack(side="left", padx=6)
        if on_block:
            ttk.Button(files, text="Block and delete file…",
                       command=lambda: self.current and self.on_block(self.current["id"])).pack(side="right")

        if on_refresh is None:
            self.refresh()
        else:
            self.count_label.config(text="Loading…")

    # ---------- data ----------
    def refresh(self):
        if self.on_refresh:
            self.on_refresh()
        else:
            self.apply(load_descriptions(self.cfg))

    def apply(self, descriptions: list[dict]):
        self.records = {d["id"]: d for d in descriptions}
        self.populate()

    def _matches(self, d: dict) -> bool:
        status = self.status_filter.get()
        flagged = bool((d.get("assessment") or {}).get("tier"))
        if status == "flagged":
            if not flagged:
                return False
        elif status != "All" and d.get("status") != status:
            return False
        needle = self.search.get().strip().lower()
        if not needle:
            return True
        hay = " ".join(str(d.get(k) or "") for k in ("name", "category", "summary", "description", "trigger_words"))
        return needle in hay.lower()

    def populate(self):
        selected = self.current["id"] if self.current else None
        self.tree.delete(*self.tree.get_children())
        shown = [d for d in self.records.values() if self._matches(d)]
        # High-risk entries sort last so they don't dominate the tree.
        shown.sort(key=lambda d: (d.get("status") in SET_ASIDE, tree_path(d), d["name"].lower()))
        counts: dict[str, int] = {}
        for d in shown:
            parts = tree_path(d)
            for i in range(1, len(parts) + 1):
                key = "/".join(parts[:i])
                counts[key] = counts.get(key, 0) + 1
        expand = bool(self.search.get().strip()) or self.status_filter.get() != "All"
        for d in shown:
            parent = ""
            parts = tree_path(d)
            for i in range(1, len(parts) + 1):
                key = "/".join(parts[:i])
                iid = "cat:" + key
                if not self.tree.exists(iid):
                    self.tree.insert(parent, "end", iid=iid, text=f"{parts[i - 1]}  ({counts[key]})",
                                     open=expand, tags=("category",))
                parent = iid
            status = d.get("status")
            self.tree.insert(parent, "end", iid="lora:" + d["id"], text=d["name"],
                             tags=(status,) if status in ("caution", "high_risk", "needs_review", "quarantined")
                             else (("flagged",) if (d.get("assessment") or {}).get("tier") else ()))
        self.count_label.config(text=f"{len(shown)} of {len(self.records)} LoRAs")
        if selected and self.tree.exists("lora:" + selected):
            self.tree.see("lora:" + selected)
            self.tree.selection_set("lora:" + selected)

    # ---------- details ----------
    def on_select(self, _event=None):
        sel = self.tree.selection()
        if not sel or not sel[0].startswith("lora:"):
            return
        d = self.records.get(sel[0][5:])
        if d is None or d is self.current:
            return
        self.current = d
        self.render_details(d)
        self.choices = image_choices(d)
        self.image_box.config(values=[label for label, _ in self.choices])
        if self.choices:
            self.image_box.current(0)
        else:
            self.image_var.set("")
        self.show_image()

    def render_details(self, d: dict):
        t = self.text
        t.config(state="normal")
        t.delete("1.0", "end")

        def line(text="", tag=None):
            t.insert("end", text + "\n", tag)

        line(d["name"], "title")
        line(f"{d.get('status')}  ·  {d.get('category')}", "muted")
        if d.get("status") in SET_ASIDE:
            line()
            line("QUARANTINED (manual review)" if d["status"] == "quarantined" else "CAUTION: set aside", "warn")
            line(d.get("high_risk_summary") or "")
            line()
            line("Technical detail", "h")
            line(d.get("high_risk_detail") or "", "muted")
            prompts = d.get("high_risk_prompts") or []
            line("Prompts rendered with this LoRA (images deleted)", "h")
            if not prompts:
                line("None: blocked before any images were generated.", "muted")
            if prompts:
                line(settings_text(d), "muted")
            for p in prompts:
                if p.get("flagged"):
                    flag = f"FLAGGED ({p.get('verdict')})"
                elif p.get("verdict") == "not screened":
                    flag = "not screened, discarded with its batch"
                else:
                    flag = f"passed ({p.get('verdict')})"
                size = f"{p['width']}x{p['height']}" if p.get("width") else p.get("aspect")
                line(f"{p['kind']} #{p['number']}: {flag}{' - ' + p['reason'] if p.get('reason') else ''}",
                     "warn" if p.get("flagged") else None)
                line(f"  seed {p.get('seed')} · weight {p.get('weight')} · {size}", "mono")
                line(f"  + {p['positive']}", "mono")
                line(f"  − {p['negative']}", "mono")
        else:
            if d.get("summary"):
                line(d["summary"], "summary")
            line(d.get("description") or "")
            rw = d.get("recommended_weight") or {}
            line("Usage", "h")
            line(f"Trigger words: {fmt(d.get('trigger_words') or d.get('trigger_words_detected') or [])}")
            line(f"Weight: min {rw.get('min')} / default {rw.get('default')} / max {rw.get('max')}")
            line(f"Type: {d.get('lora_type')}   NSFW: {d.get('nsfw_level')}   Confidence: {d.get('confidence')}")
            line("Metrics", "h")
            for name, value in {**(d.get("metrics") or {}), **(d.get("extra_metrics") or {})}.items():
                line(f"{name}: {fmt(value)}")
            if d.get("scores"):
                line("Scores", "h")
                line("   ".join(f"{k.replace('_', ' ')}: {v}/10" for k, v in d["scores"].items()))
            for key, title in (("strengths", "Strengths"), ("weaknesses", "Weaknesses"), ("usage_tips", "Usage tips")):
                if d.get(key):
                    line(title, "h")
                    for item in d[key]:
                        line(f"•  {item}")
            if d.get("category_reason"):
                line("Why this category", "h")
                line(d["category_reason"])
        flag = d.get("assessment") or {}
        if d.get("status") not in SET_ASIDE and flag.get("tier"):
            line("Safety flag", "h")
            line(flag.get("summary") or flag["tier"], "warn" if flag["tier"] == "caution" else None)
        if d.get("discarded_images"):
            line("Discarded by the screener", "h")
            line(f"{len(d['discarded_images'])} generated image(s) were flagged and discarded (never saved); "
                 "the LoRA is approved, so the rest were kept.", "muted")
        if d.get("status") not in SET_ASIDE and d.get("render_settings"):
            line("Generation settings", "h")
            line(settings_text(d), "muted")
        line("Files", "h")
        line(f"Previously in: {d.get('previous_folder')}", "muted")
        line(f"LoRA file: {d.get('lora_file')}", "muted")
        line(f"Source: {d.get('source_file')}", "muted")
        civ = d.get("civitai")
        if civ:
            sizes = ", ".join(f"{k} ×{v}" for k, v in (civ.get("example_resolutions") or {}).items())
            line(f"Civitai: {civ.get('model')} ({civ.get('version')})  ·  {civ.get('url')}", "muted")
            line(f"Civitai example resolutions: {sizes or 'none'}", "muted")
        t.config(state="disabled")
        t.yview_moveto(0)

    # ---------- images ----------
    def selected_image(self) -> dict | None:
        idx = self.image_box.current()
        return self.choices[idx][1] if 0 <= idx < len(self.choices) else None

    def step_image(self, delta: int):
        if self.choices:
            self.image_box.current((self.image_box.current() + delta) % len(self.choices))
            self.show_image()

    def on_resize(self, _event=None):
        if self._resize_job:
            self.image_label.after_cancel(self._resize_job)
        self._resize_job = self.image_label.after(150, self.show_image)

    def show_image(self):
        self._resize_job = None
        img = self.selected_image()
        if not img:
            self._photo = None
            text = "No images: blocked or not rendered" if self.current else "Select a LoRA"
            self.image_label.config(image="", text=text)
            self.image_caption.config(text="")
            return
        path = Path(img["path"])
        if not path.exists():
            self._photo = None
            self.image_label.config(image="", text=f"Missing file: {path}")
        else:
            w = max(self.image_label.winfo_width() - 8, 100)
            h = max(self.image_label.winfo_height() - 8, 100)
            with Image.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail((w, h))
                self._photo = ImageTk.PhotoImage(im)  # keep a reference or Tk drops it
            self.image_label.config(image=self._photo, text="")
        parts = [f"weight {img.get('lora_weight')}", f"seed {img.get('seed')}"]
        if img.get("width"):
            parts.append(f"{img['width']}x{img['height']}")
        if img.get("notes"):
            parts.append(img["notes"])
        self.image_caption.config(text=f"{img.get('prompt')}\n" + "  ·  ".join(str(p) for p in parts))

    # ---------- open files ----------
    def open_image(self):
        img = self.selected_image()
        if img and Path(img["path"]).exists():
            os.startfile(img["path"])

    def open_description(self):
        if self.current:
            md = Path(self.current["_json_path"]).with_suffix(".md")
            os.startfile(md if md.exists() else self.current["_json_path"])

    def show_lora_file(self):
        if self.current and self.current.get("lora_file") and Path(self.current["lora_file"]).exists():
            subprocess.Popen(f'explorer /select,"{self.current["lora_file"]}"')

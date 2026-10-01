"""Review tab: manually review set-aside LoRAs, approve them for testing, run test renders,
reprocess them, and block (blocklist + delete) any LoRA. All work goes through actions."""
import io
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

from PIL import Image, ImageTk

from . import actions, blocklist, reviews, safety
from .civitai import BUCKET_NAMES
from .store import load_state

FILTERS = ["All", "unreviewed", "reviewable", "caution", "held", "set aside", "approved", "quarantined", "blocked"]
SIZES = ["1024x1024", "832x1216", "896x1152", "1216x832", "1152x896", "768x1344", "1344x768", "640x1536", "1536x640"]
assert set(SIZES) == BUCKET_NAMES


class ReviewTab:
    def __init__(self, parent: ttk.Frame, app):
        self.app, self.cfg = app, app.cfg
        self.items: dict[str, dict] = {}
        self.current: dict | None = None
        self.recorded: list[dict] = []
        self._photos: list = []

        bar = ttk.Frame(parent, padding=(0, 0, 0, 8))
        bar.pack(fill="x")
        ttk.Label(bar, text="Show:").pack(side="left")
        self.filter = tk.StringVar(value="All")
        box = ttk.Combobox(bar, textvariable=self.filter, values=FILTERS, width=12, state="readonly")
        box.pack(side="left", padx=4)
        box.bind("<<ComboboxSelected>>", lambda _e: self.populate())
        ttk.Button(bar, text="Refresh", command=self.refresh).pack(side="left", padx=6)
        ttk.Button(bar, text="Blocklist log", command=self.show_blocklist_log).pack(side="left")
        self.count_label = ttk.Label(bar, foreground="gray")
        self.count_label.pack(side="right")

        panes = ttk.PanedWindow(parent, orient="horizontal")
        panes.pack(fill="both", expand=True)

        left = ttk.Frame(panes)
        self.tree = ttk.Treeview(left, columns=("status", "review"), show="tree headings", selectmode="browse")
        self.tree.heading("#0", text="LoRA")
        self.tree.heading("status", text="Status")
        self.tree.heading("review", text="Review")
        self.tree.column("#0", width=220)
        self.tree.column("status", width=165, stretch=False)
        self.tree.column("review", width=85, stretch=False)
        scroll = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self.on_select)
        self.tree.tag_configure("quarantined", foreground="#7a0010")
        self.tree.tag_configure("caution", foreground="#b00020")
        self.tree.tag_configure("reviewable", foreground="#9a6700")
        self.tree.tag_configure("approved", foreground="#1d6b2f")
        self.tree.tag_configure("blocked", foreground="gray")
        panes.add(left, weight=1)

        right = ttk.Frame(panes)
        panes.add(right, weight=3)

        self.text = tk.Text(right, wrap="word", state="disabled", font=("Segoe UI", 10), height=12,
                            padx=10, pady=8, relief="flat")
        self.text.pack(fill="both", expand=True)
        self.text.tag_configure("title", font=("Segoe UI", 14, "bold"), spacing3=4)
        self.text.tag_configure("h", font=("Segoe UI", 10, "bold"), spacing1=8, spacing3=2)
        self.text.tag_configure("muted", foreground="gray")
        self.text.tag_configure("warn", foreground="#b00020", font=("Segoe UI", 10, "bold"))

        actions_row = ttk.Frame(right, padding=(0, 6))
        actions_row.pack(fill="x")
        style = ttk.Style()
        style.configure("Danger.TButton", foreground="#b00020")
        self.approve_btn = ttk.Button(actions_row, text="Approve…", command=self.approve)
        self.revoke_btn = ttk.Button(actions_row, text="Revoke", command=self.revoke)
        self.reprocess_btn = ttk.Button(actions_row, text="Reprocess", command=self.reprocess)
        self.quarantine_btn = ttk.Button(actions_row, text="Quarantine…", style="Danger.TButton", command=self.quarantine)
        self.release_btn = ttk.Button(actions_row, text="Release…", command=self.release)
        self.block_btn = ttk.Button(actions_row, text="Block & delete…", style="Danger.TButton",
                                    command=lambda: self.current and self.app.confirm_block(self.current["id"]))
        for b in (self.approve_btn, self.revoke_btn, self.quarantine_btn, self.release_btn, self.reprocess_btn):
            b.pack(side="left", padx=(0, 6))
        self.block_btn.pack(side="right")

        # ---- test bench ----
        bench = ttk.LabelFrame(right, text="Test render", padding=8)
        bench.pack(fill="x")
        bench.columnconfigure(1, weight=1)
        ttk.Label(bench, text="Start from:").grid(row=0, column=0, sticky="w")
        self.source_var = tk.StringVar(value="Custom prompt")
        self.source_box = ttk.Combobox(bench, textvariable=self.source_var, state="readonly")
        self.source_box.grid(row=0, column=1, columnspan=5, sticky="ew", pady=2)
        self.source_box.bind("<<ComboboxSelected>>", lambda _e: self.load_recorded())
        ttk.Label(bench, text="Prompt:").grid(row=1, column=0, sticky="nw")
        self.prompt = tk.Text(bench, height=3, wrap="word", font=("Consolas", 9))
        self.prompt.grid(row=1, column=1, columnspan=5, sticky="ew", pady=2)
        ttk.Label(bench, text="Extra negative:").grid(row=2, column=0, sticky="w")
        self.neg = tk.StringVar()
        ttk.Entry(bench, textvariable=self.neg).grid(row=2, column=1, columnspan=5, sticky="ew", pady=2)

        opts = ttk.Frame(bench)
        opts.grid(row=3, column=0, columnspan=6, sticky="w", pady=(4, 0))
        self.weight = tk.StringVar(value="0.9")
        self.seed = tk.StringVar()
        self.size = tk.StringVar(value="1024x1024")
        self.count = tk.StringVar(value="2")
        self.save = tk.BooleanVar(value=False)
        for label, widget in (("Weight", ttk.Spinbox(opts, from_=-3, to=3, increment=0.1, textvariable=self.weight, width=6)),
                              ("Seed (blank = random)", ttk.Entry(opts, textvariable=self.seed, width=12)),
                              ("Size", ttk.Combobox(opts, textvariable=self.size, values=SIZES, width=10, state="readonly")),
                              ("Images", ttk.Spinbox(opts, from_=1, to=4, textvariable=self.count, width=4))):
            ttk.Label(opts, text=label + ":").pack(side="left", padx=(0, 3))
            widget.pack(side="left", padx=(0, 12))
        ttk.Checkbutton(opts, text="Save passing images", variable=self.save).pack(side="left")
        self.test_btn = ttk.Button(bench, text="Render test", command=self.render_test)
        self.test_btn.grid(row=4, column=0, sticky="w", pady=(6, 0))
        self.bench_note = ttk.Label(bench, foreground="gray", wraplength=600, justify="left")
        self.bench_note.grid(row=4, column=1, columnspan=5, sticky="w", pady=(6, 0))

        self.results = ttk.Frame(right, padding=(0, 6))
        self.results.pack(fill="x")

        self.count_label.config(text="Loading…")  # the app loads the data in the background

    # ---------- data ----------
    def refresh(self):
        self.app.refresh_all()

    def apply(self, review_items: list[dict], blocked_entries: list[dict]):
        items = {i["id"]: i for i in review_items}
        known = {i.get("sha256") for i in items.values()}
        for e in blocked_entries:  # blocked hashes with no record here
            if e["sha256"] not in known:
                items["sha:" + e["sha256"]] = {"id": "sha:" + e["sha256"], "name": e.get("filename"), "rel": e.get("source_path"),
                                               "status": "blocked", "sha256": e["sha256"], "approval": None,
                                               "quarantine": None, "tier": None, "held": False,
                                               "summary": f"Blocked {e.get('blocked_at')}: {e.get('reason')}"}
        self.items = items
        self.populate()

    def populate(self):
        selected = self.current["id"] if self.current else None
        self.tree.delete(*self.tree.get_children())
        f = self.filter.get()
        shown = [i for i in self.items.values() if self._matches(i, f)]
        order = {"quarantined": 0, "caution": 1, "reviewable": 2, "approved": 3, "blocked": 4}
        for i in sorted(shown, key=lambda i: (order.get(self.label(i), 2), (i["name"] or "").lower())):
            label = self.label(i)
            shown_status = label + (" (held)" if i.get("held") and label == "caution" else "")
            if i["status"] == "caution" and label == "caution":
                shown_status += " · set aside"
            self.tree.insert("", "end", iid=i["id"], text=i["name"],
                             values=(shown_status, "approved" if i["approval"] else ("quarantined" if i.get("quarantine") else "")),
                             tags=(label,))
        self.count_label.config(text=f"{len(shown)} shown · {sum(1 for i in self.items.values() if i['approval'])} approved")
        if selected and self.tree.exists(selected):
            self.tree.selection_set(selected)
        else:
            self.current = None
            self.render_details()

    @staticmethod
    def label(i: dict) -> str:
        """The tier shown for an item: blocked > quarantined > caution > reviewable > approved."""
        if i["status"] == "blocked":
            return "blocked"
        if i.get("quarantine") or i["status"] == "quarantined":
            return "quarantined"
        if i["status"] == "caution" or i.get("tier") == safety.CAUTION:
            return "caution"
        if i.get("tier") == safety.REVIEWABLE:
            return "reviewable"
        return "approved" if i.get("approval") else (i.get("tier") or i["status"])

    def _matches(self, i: dict, f: str) -> bool:
        label = self.label(i)
        return (f == "All" or f == label or (f == "approved" and bool(i["approval"]))
                or (f == "unreviewed" and label in ("reviewable", "caution") and not i["approval"])
                or (f == "held" and i.get("held")) or (f == "set aside" and i["status"] == "caution"))

    def on_select(self, _e=None):
        sel = self.tree.selection()
        new = self.items.get(sel[0]) if sel else None
        if (new or {}).get("id") != (self.current or {}).get("id"):
            for w in self.results.winfo_children():  # results belong to the previous LoRA
                w.destroy()
        self.current = new
        self.render_details()

    def eligibility(self) -> tuple[bool, str]:
        i = self.current
        if not i:
            return False, "Select a LoRA."
        if i["status"] == "blocked":
            return False, "Blocked: the file was deleted."
        if i.get("quarantine") or i["status"] == "quarantined":
            return False, "Quarantined by manual review: it can't be rendered. Release it or block it."
        if (i["status"] == "caution" or i.get("held")) and not i["approval"]:
            return False, ("Held (states an age under 18): approve it to render it." if i.get("held") else
                           "Set aside because generated images looked childlike: approve it to test it again.")
        return True, ""

    def render_details(self):
        t = self.text
        t.config(state="normal")
        t.delete("1.0", "end")
        i = self.current
        if not i:
            t.insert("end", "Select a LoRA on the left.", "muted")
            t.config(state="disabled")
            self._update_buttons()
            return
        t.insert("end", (i["name"] or "") + "\n", "title")
        t.insert("end", f"{i['status']}  ·  {i['rel']}\nSHA-256 {i.get('sha256') or 'unknown'}\n", "muted")
        label = self.label(i)
        if label in ("quarantined", "caution", "reviewable"):
            t.insert("end", label.upper() + (" · HELD" if i.get("held") else "") + "\n",
                     "warn" if label != "reviewable" else "h")
        if i.get("quarantine"):
            q = i["quarantine"]
            t.insert("end", f"Quarantined {q['quarantined_at']}: {q.get('notes') or '(no notes)'}\n")
        if i.get("summary"):
            t.insert("end", i["summary"] + "\n")
        state = None if i["id"].startswith("sha:") else load_state(self.cfg, i["id"])
        assessment = i.get("assessment") or (state or {}).get("safety", {}).get("assessment") or {}
        if assessment.get("reasons"):
            t.insert("end", "Evidence\n", "h")
            for r in assessment["reasons"]:
                t.insert("end", f"•  {r}\n")
        if assessment and not i.get("quarantine"):
            mode = safety.render_mode(assessment, True)
            t.insert("end", "If approved\n", "h")
            t.insert("end", ("Explicit LoRA: renders in reviewed-adult mode (stronger adult cues, weighted CSAM "
                             "negatives). The screener flags minors only on clear evidence; images judged childlike "
                             "AND sexual/suggestive are discarded unseen, the rest kept. The approval stays until "
                             "you revoke it.\n") if mode == safety.REVIEWED_ADULT else
                     ("Non-explicit: renders in forced-SFW mode (rating_safe, sexual tags stripped, strong NSFW/CSAM "
                      "negatives). Childlike non-sexual images are kept; childlike + suggestive/explicit images are "
                      "discarded unseen. The approval stays until you revoke it.\n"))
        if i["approval"]:
            a = i["approval"]
            t.insert("end", "Manual review\n", "h")
            t.insert("end", f"Approved {a['approved_at']}: {a.get('notes') or '(no notes)'}\n")
        hist = reviews.history(self.cfg, i.get("sha256")) if i.get("sha256") else []
        if hist:
            t.insert("end", "History\n", "h")
            for e in hist[-12:]:
                extra = e.get("reason") or e.get("notes") or (f"passed={e['passed']} ({e.get('mode')})" if "passed" in e else "")
                t.insert("end", f"{e['time']}  {e['event']}  {extra}\n", "muted")
        t.config(state="disabled")

        self.recorded = (state or {}).get("safety", {}).get("high_risk_prompts") or []
        labels = ["Custom prompt"] + [f"{p['kind']} #{p['number']} · seed {p['seed']} · "
                                      f"{'FLAGGED' if p.get('flagged') else p.get('verdict')}" for p in self.recorded]
        self.source_box.config(values=labels)
        self.source_var.set("Custom prompt")
        self._update_buttons()

    def _update_buttons(self):
        i = self.current
        ok, why = self.eligibility()
        quarantined = bool(i) and (bool(i.get("quarantine")) or i["status"] == "quarantined")
        can_approve = bool(i) and i["status"] != "blocked" and not i["approval"] and not quarantined
        self.approve_btn.config(state="normal" if can_approve else "disabled")
        self.quarantine_btn.config(state="normal" if i and i["status"] != "blocked" and not quarantined else "disabled")
        self.release_btn.config(state="normal" if quarantined and i["status"] != "blocked" else "disabled")
        self.revoke_btn.config(state="normal" if i and i["approval"] else "disabled")
        self.reprocess_btn.config(state="normal" if ok else "disabled")
        self.block_btn.config(state="normal" if i and i["status"] != "blocked" else "disabled")
        self.test_btn.config(state="normal" if ok else "disabled")
        self.bench_note.config(text=why or "Images are screened before they're shown; flagged images are never displayed.")

    def load_recorded(self):
        idx = self.source_box.current()
        if idx <= 0:
            return
        p = self.recorded[idx - 1]
        self.prompt.delete("1.0", "end")
        self.prompt.insert("1.0", p["positive"])
        self.seed.set(str(p["seed"]))
        self.weight.set(str(p["weight"] or 0.9))
        if p.get("width"):
            size = f"{p['width']}x{p['height']}"
            self.size.set(size if size in SIZES else "1024x1024")

    # ---------- actions ----------
    def approve(self):
        i = self.current
        notes = simpledialog.askstring(
            "Approve",
            f"Approve {i['name']}?\n\nThis is logged and stays until you revoke it. It returns the LoRA to the "
            "normal flow (set-aside LoRAs are queued for Continue/Reprocess). Every image is still screened: "
            "flagged images are discarded unseen and the rest are kept.\n\nReview notes (why it's OK):",
            parent=self.tree)
        if notes is None:
            return
        self.app.start_task("Approve", actions.approve, self.cfg, i["id"], notes.strip(), on_result=lambda _r: self.app.refresh_all())

    def quarantine(self):
        i = self.current
        notes = simpledialog.askstring(
            "Quarantine", f"Quarantine {i['name']}?\n\nIts images are deleted and it will never be rendered "
            "(until released). This is logged.\n\nReview notes (why):", parent=self.tree)
        if notes is None:
            return
        self.app.start_task("Quarantine", actions.quarantine, self.cfg, i["id"], notes.strip(),
                            on_result=lambda _r: self.app.refresh_all())

    def release(self):
        i = self.current
        notes = simpledialog.askstring(
            "Release from quarantine", f"Release {i['name']} from quarantine?\n\nIt will be processed again on the "
            "next run under the normal rules. This is logged.\n\nNotes (why):", parent=self.tree)
        if notes is None:
            return
        self.app.start_task("Release", actions.release, self.cfg, i["id"], notes.strip(),
                            on_result=lambda _r: self.app.refresh_all())

    def revoke(self):
        if messagebox.askyesno("Revoke approval", f"Revoke the manual review approval for {self.current['name']}?"):
            self.app.start_task("Revoke", actions.revoke, self.cfg, self.current["id"], on_result=lambda _r: self.app.refresh_all())

    def reprocess(self):
        i = self.current
        if messagebox.askyesno("Reprocess", f"Run {i['name']} through the full pipeline again?\n\n"
                               "If every image passes screening it is classified normally; otherwise it is set aside (caution)."):
            self.app.start_task("Reprocess", actions.reprocess, self.cfg, i["id"],
                                should_stop=self.app.stop_event.is_set, on_result=lambda _r: self.app.refresh_all())

    def render_test(self):
        i = self.current
        try:
            params = {"positive": self.prompt.get("1.0", "end").strip(), "negative_extra": self.neg.get().strip(),
                      "weight": float(self.weight.get()), "seed": int(self.seed.get()) if self.seed.get().strip() else None,
                      "aspect": self.size.get(), "count": int(self.count.get()), "save": self.save.get()}
        except ValueError:
            messagebox.showerror("Test render", "Weight, seed and images must be numbers.")
            return
        if not params["positive"]:
            messagebox.showerror("Test render", "Enter a prompt (or pick a recorded one).")
            return
        self.app.start_task("Test render", actions.test, self.cfg, i["id"], on_result=self.show_results, **params)

    def show_results(self, out: dict):
        for w in self.results.winfo_children():
            w.destroy()
        self._photos = []
        shown = sum(1 for r in out["results"] if r["png"])
        if out["passed"]:
            head = f"Mode: {out['mode']}.  All images passed screening."
        elif out.get("approved"):
            head = (f"Mode: {out['mode']}.  {len(out['results']) - shown} image(s) flagged and discarded unseen; "
                    f"showing the {shown} that passed. The approval is kept.")
        else:
            head = f"Mode: {out['mode']}.  Screening failed: nothing is shown and the batch was discarded."
        if out.get("error"):
            head += f"  ({out['error']})"
        ttk.Label(self.results, text=head, foreground="black" if out["passed"] else "#b00020",
                  wraplength=700, justify="left").pack(anchor="w")
        if out.get("high_risk_summary"):
            ttk.Label(self.results, text=out["high_risk_summary"], foreground="#b00020", wraplength=700,
                      justify="left").pack(anchor="w", pady=(2, 4))
        row = ttk.Frame(self.results)
        row.pack(anchor="w")
        for r in out["results"]:
            cell = ttk.Frame(row, padding=4)
            cell.pack(side="left", anchor="n")
            if r["png"]:  # only ever present when the whole batch passed
                im = Image.open(io.BytesIO(r["png"])).convert("RGB")
                im.thumbnail((220, 220))
                photo = ImageTk.PhotoImage(im)
                self._photos.append(photo)
                ttk.Label(cell, image=photo).pack()
            else:
                ttk.Label(cell, text="(not shown)", width=28, anchor="center", foreground="gray").pack(pady=40)
            ttk.Label(cell, text=f"seed {r['seed']} · {r['verdict']}/{r['sexual']}\n{r.get('reason') or ''}",
                      wraplength=220, foreground="gray", justify="left").pack()
        self.app.refresh_all()

    def show_blocklist_log(self):
        win = tk.Toplevel(self.tree)
        win.title("Blocklist log")
        win.geometry("900x400")
        txt = tk.Text(win, wrap="none", font=("Consolas", 9))
        txt.pack(fill="both", expand=True)
        txt.insert("1.0", blocklist.read_log(self.cfg) or "Nothing blocked yet.")
        txt.config(state="disabled")

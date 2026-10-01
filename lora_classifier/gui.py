"""Simple desktop UI: a Start button with the run parameters, a live log, and the
secondary tools (preview, status, rebuild). All work is done by lora_classifier.actions."""
import os
import queue
import sys
import threading


def _pin_tcl_library():
    """Point Tcl/Tk at the base Python's libraries. From a venv, Tcl sometimes fails to locate
    init.tcl on its own ("Can't find a usable init.tcl"); being explicit avoids that."""
    base = os.path.join(sys.base_prefix, "tcl")
    for var, sub in (("TCL_LIBRARY", "tcl8.6"), ("TK_LIBRARY", "tk8.6")):
        path = os.path.join(base, sub)
        if not os.environ.get(var) and os.path.isfile(os.path.join(path, "init.tcl" if var == "TCL_LIBRARY" else "tk.tcl")):
            os.environ[var] = path


_pin_tcl_library()

import tkinter as tk  # noqa: E402 - after the Tcl paths are pinned
from tkinter import messagebox, scrolledtext, simpledialog, ttk

from . import actions, pipeline
from .inspector import Inspector
from .review_tab import ReviewTab
from .store import load_state
from .comfy import ComfyError
from .config import Config
from .llm import LLMError

MAX_LOG_LINES = 5000


class App:
    def __init__(self, root: tk.Tk, cfg: Config):
        self.root, self.cfg = root, cfg
        self.events: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.loader: threading.Thread | None = None  # background refresh of the tabs' data
        self._reload_again = False
        self.closing = False

        root.title("LoRA Classifier")
        root.geometry("1100x780")
        root.minsize(720, 520)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        tabs = ttk.Notebook(root)
        tabs.pack(fill="both", expand=True, padx=6, pady=6)
        outer = ttk.Frame(tabs, padding=12)
        inspector_tab = ttk.Frame(tabs, padding=12)
        review_tab = ttk.Frame(tabs, padding=12)
        tabs.add(outer, text="Run")
        tabs.add(inspector_tab, text="Inspector")
        tabs.add(review_tab, text="Review")
        self.tabs = tabs
        self.inspector = Inspector(inspector_tab, cfg, on_block=self.confirm_block, on_refresh=self.refresh_all)
        self.review = ReviewTab(review_tab, self)

        # ---- run parameters ----
        params = ttk.LabelFrame(outer, text="Run settings", padding=10)
        params.pack(fill="x")
        params.columnconfigure(1, weight=1)

        self.match = tk.StringVar()
        self.limit = tk.StringVar(value="0")
        self.probes = tk.StringVar(value=str(cfg.probe["count"]))
        self.examples = tk.BooleanVar(value=True)
        self.redo = tk.BooleanVar(value=False)

        ttk.Label(params, text="Only LoRAs whose path contains:").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(params, textvariable=self.match).grid(row=0, column=1, columnspan=2, sticky="ew", pady=3)
        ttk.Label(params, text="(blank = all, e.g. styles/anime)", foreground="gray").grid(
            row=1, column=1, columnspan=2, sticky="w")

        ttk.Label(params, text="Max LoRAs this run:").grid(row=2, column=0, sticky="w", pady=3)
        ttk.Spinbox(params, from_=0, to=10000, textvariable=self.limit, width=8).grid(row=2, column=1, sticky="w", pady=3)
        ttk.Label(params, text="(0 = no limit)", foreground="gray").grid(row=2, column=2, sticky="w")

        ttk.Label(params, text="Probe images per LoRA:").grid(row=3, column=0, sticky="w", pady=3)
        ttk.Combobox(params, textvariable=self.probes, values=["4", "8", "16"], width=6, state="readonly").grid(
            row=3, column=1, sticky="w", pady=3)

        ttk.Checkbutton(params, text="Generate example images and evaluate them", variable=self.examples).grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Checkbutton(params, text="Reprocess LoRAs that are already finished", variable=self.redo).grid(
            row=5, column=0, columnspan=3, sticky="w")

        # ---- start / stop / progress ----
        controls = ttk.Frame(outer, padding=(0, 10))
        controls.pack(fill="x")
        style = ttk.Style()
        style.configure("Start.TButton", font=("Segoe UI", 11, "bold"), padding=(18, 6))
        self.start_btn = ttk.Button(controls, text="Start", style="Start.TButton", command=self.start_run)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(controls, text="Stop", command=self.request_stop, state="disabled")
        self.stop_btn.pack(side="left", padx=8)
        self.progress_label = ttk.Label(controls, text="Idle")
        self.progress_label.pack(side="left", padx=8)
        self.progress = ttk.Progressbar(outer, mode="determinate")
        self.progress.pack(fill="x")
        self.library_label = ttk.Label(outer, foreground="gray")
        self.library_label.pack(fill="x", pady=(4, 0))

        # ---- secondary tools ----
        tools = ttk.Frame(outer, padding=(0, 8))
        tools.pack(fill="x")
        self.tool_buttons = [
            ttk.Button(tools, text="Preview (no rendering)", command=self.start_scan),
            ttk.Button(tools, text="Status", command=lambda: self.start_task("Status", actions.status, self.cfg, True)),
            ttk.Button(tools, text="Rebuild output folders", command=self.start_rebuild),
        ]
        for b in self.tool_buttons:
            b.pack(side="left", padx=(0, 6))
        ttk.Button(tools, text="Open output folder", command=self.open_output).pack(side="right")
        ttk.Button(tools, text="Open catalogue", command=self.open_catalogue).pack(side="right", padx=6)

        # ---- log ----
        self.log = scrolledtext.ScrolledText(outer, height=18, wrap="word", state="disabled",
                                             font=("Consolas", 9))
        self.log.pack(fill="both", expand=True)
        self.log.tag_configure("warn", foreground="#b00020")

        footer = f"ComfyUI: {cfg.comfy['url']}   ·   Model: {cfg.deepinfra['model']}   ·   Output: {cfg.output_dir}"
        ttk.Label(outer, text=footer, foreground="gray").pack(fill="x", pady=(6, 0))

        pipeline.set_log_sink(lambda line: self.events.put(("log", line)))
        self.match.trace_add("write", lambda *_: self._schedule_progress())
        self.redo.trace_add("write", lambda *_: self.apply_progress(self._last_progress))
        self._progress_job = None
        self._last_progress = None
        self.library_label.config(text="Loading…")
        self.refresh_all()
        self.root.after(100, self.poll)

    def _schedule_progress(self):
        if self._progress_job:
            self.root.after_cancel(self._progress_job)
        self._progress_job = self.root.after(400, self.refresh_progress)

    # ---------- background work ----------
    def busy(self) -> bool:
        return self.worker is not None and self.worker.is_alive()

    def start_task(self, label: str, fn, *args, on_result=None, **kwargs):
        """Run fn in the background. on_result(result) is called on the UI thread if it succeeds."""
        if self.busy():
            messagebox.showinfo("LoRA Classifier", "Another task is still running. Wait for it or press Stop.")
            return
        self.stop_event.clear()
        self.set_running(True, run=fn in (actions.run, actions.reprocess))
        self.progress_label.config(text=f"{label}…")

        def work():
            try:
                result = fn(*args, **kwargs)
                if on_result:
                    self.events.put(("result", (on_result, result)))
                stopped = isinstance(result, dict) and result.get("stopped")
                self.events.put(("done", "Stopped. Press Start to resume." if stopped else f"{label} finished"))
            except (ComfyError, LLMError, PermissionError) as e:
                self.events.put(("error", str(e)))
            except Exception as e:  # noqa: BLE001 - surface anything unexpected in the UI
                self.events.put(("error", f"{type(e).__name__}: {e}"))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def start_run(self):
        try:
            limit = int(self.limit.get() or 0)
        except ValueError:
            messagebox.showerror("LoRA Classifier", "Max LoRAs must be a whole number.")
            return
        self.progress.config(value=0, maximum=1)
        self.start_task("Running", actions.run, self.cfg,
                        match=self.match.get().strip() or None, limit=limit or None,
                        probes=int(self.probes.get()), redo=self.redo.get(), no_examples=not self.examples.get(),
                        should_stop=self.stop_event.is_set,
                        on_progress=lambda n, total, name: self.events.put(("progress", (n, total, name))))

    def start_scan(self):
        self.start_task("Preview", actions.scan, self.cfg, self.match.get().strip() or None)

    def start_rebuild(self):
        if messagebox.askyesno("Rebuild output folders",
                               "Recreate output/loras and the descriptions from saved results?\n"
                               "(The original LoRA folder is never touched.)"):
            self.start_task("Rebuild", actions.rebuild, self.cfg)

    def request_stop(self):
        self.stop_event.set()
        self.stop_btn.config(state="disabled")
        self.progress_label.config(text="Stopping after the current image…")

    def set_running(self, running: bool, run: bool = False):
        state = "disabled" if running else "normal"
        self.start_btn.config(state=state)
        for b in self.tool_buttons:
            b.config(state=state)
        self.stop_btn.config(state="normal" if running and run else "disabled")

    # ---------- UI updates (main thread only) ----------
    def append_log(self, text: str):
        self.log.config(state="normal")
        tag = "warn" if any(k in text for k in ("CAUTION", "QUARANTINED", "BLOCKED", "ERROR")) or text.startswith("error") else None
        self.log.insert("end", text + "\n", tag)
        excess = int(self.log.index("end-1c").split(".")[0]) - MAX_LOG_LINES
        if excess > 0:
            self.log.delete("1.0", f"{excess + 1}.0")
        self.log.see("end")
        self.log.config(state="disabled")

    def poll(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "log":
                    self.append_log(payload)
                elif kind == "progress":
                    n, total, name = payload
                    self.progress.config(maximum=max(total, 1), value=n)
                    self.progress_label.config(text=f"LoRA {n + 1}/{total}: {name}" if name else f"{n}/{total} done")
                elif kind == "result":
                    callback, result = payload
                    callback(result)
                elif kind == "refreshed":
                    self.inspector.apply(payload["descriptions"])
                    self.review.apply(payload["review"], payload["blocked"])
                    self.apply_progress(payload["progress"])
                    if self._reload_again:
                        self.refresh_all()
                elif kind == "progress_data":
                    self.apply_progress(payload)
                elif kind == "refresh_error":
                    self.library_label.config(text=f"(couldn't load: {payload})")
                elif kind == "done":
                    self.set_running(False)
                    self.progress_label.config(text=payload)
                    self.refresh_all()
                elif kind == "error":
                    self.set_running(False)
                    self.refresh_all()
                    self.progress_label.config(text="Stopped with an error")
                    self.append_log(f"error: {payload}")
                    messagebox.showerror("LoRA Classifier", payload)
        except queue.Empty:
            pass
        if self.closing and not self.busy():
            self.root.destroy()
            return
        self.root.after(100, self.poll)

    # ---------- safety actions shared by the tabs ----------
    def loading(self) -> bool:
        return self.loader is not None and self.loader.is_alive()

    def refresh_all(self):
        """Reload the Inspector, Review and progress data in the background, then redraw.
        Nothing slow ever runs on the UI thread. Requests made while a reload is running are
        coalesced into one more reload afterwards."""
        if self.loading():
            self._reload_again = True
            return
        self._reload_again = False
        cfg, match = self.cfg, self.match.get().strip() or None
        from . import blocklist
        from .inspector import load_descriptions

        def load():
            try:
                self.events.put(("refreshed", {"review": actions.review_items(cfg),
                                               "blocked": blocklist.entries(cfg),
                                               "descriptions": load_descriptions(cfg),
                                               "progress": actions.progress(cfg, match)}))
            except Exception as e:  # noqa: BLE001 - shown in the UI instead of crashing it
                self.events.put(("refresh_error", f"{type(e).__name__}: {e}"))

        self.loader = threading.Thread(target=load, daemon=True)
        self.loader.start()

    def refresh_progress(self):
        """Recount progress (e.g. after the path filter changed) without blocking the UI."""
        cfg, match = self.cfg, self.match.get().strip() or None

        def load():
            try:
                self.events.put(("progress_data", actions.progress(cfg, match)))
            except Exception as e:  # noqa: BLE001
                self.events.put(("refresh_error", f"{type(e).__name__}: {e}"))
        threading.Thread(target=load, daemon=True).start()

    def apply_progress(self, p: dict | None):
        """Show how far the library has got, and make Start say it will continue."""
        if not p:
            return
        self._last_progress = p
        scope = f" matching '{self.match.get().strip()}'" if self.match.get().strip() else ""
        parts = [f"{p['processed']} of {p['total']} LoRAs{scope} processed ({p['finished']} finished, "
                 f"{p['set_aside']} set aside)"]
        if p["remaining"]:
            detail = [f"{p['not_started']} not started"] if p["not_started"] else []
            if p.get("queued"):
                detail.append(f"{p['queued']} queued for another pass")
            if p["interrupted"]:
                detail.append(f"{p['interrupted']} interrupted")
            if p["errors"]:
                detail.append(f"{p['errors']} with errors")
            parts.append(f"{p['remaining']} remaining ({', '.join(detail)}). Start continues with these.")
        else:
            parts.append("nothing left to do (tick 'Reprocess' to redo finished LoRAs)")
        self.library_label.config(text=" · ".join(parts))
        if not self.busy():
            redo = self.redo.get()
            self.start_btn.config(text="Start" if redo or not p["processed"] or not p["remaining"]
                                  else f"Continue ({p['remaining']} left)")

    def confirm_block(self, lora_id: str):
        """Block = blocklist the hash and permanently delete the .safetensors file."""
        if self.busy():
            messagebox.showinfo("LoRA Classifier", "Wait for the running task to finish first.")
            return
        state = load_state(self.cfg, lora_id)
        if not state:
            return
        meta = state["meta"]
        if not messagebox.askyesno(
                "Block and delete LoRA",
                f"Permanently delete this LoRA and add it to the blocklist?\n\n{meta['source_path']}\n\n"
                "The .safetensors file is deleted from your LoRA folder (not moved to the Recycle Bin), along "
                "with this tool's copy, images and description of it. If the same file ever shows up again, "
                "it will be deleted automatically before the next run. This can't be undone.",
                icon="warning", default="no"):
            return
        reason = simpledialog.askstring("Block reason", "Reason (saved in the blocklist and its log):",
                                        parent=self.root)
        if reason is None:
            return
        self.start_task("Block", actions.block, self.cfg, lora_id, reason.strip() or "blocked manually",
                        on_result=lambda _r: self.refresh_all())

    # ---------- misc ----------
    def open_catalogue(self):
        index = self.cfg.catalog_dir / "index.md"
        if index.exists():
            os.startfile(index)
        else:
            messagebox.showinfo("LoRA Classifier", "No catalogue yet. Run the classifier first.")

    def open_output(self):
        out = self.cfg.assert_writable(self.cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        os.startfile(out)

    def on_close(self):
        if self.busy():
            if not messagebox.askyesno("LoRA Classifier",
                                       "A task is running. Stop it and close when the current image finishes?"):
                return
            self.request_stop()
            self.closing = True
            self.root.title("LoRA Classifier (closing after the current image…)")
        else:
            self.root.destroy()


def launch(config_path=None):
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # crisp text on high-DPI displays
    except (AttributeError, OSError):
        pass
    root = tk.Tk()
    try:
        cfg = Config(config_path)
    except Exception as e:  # noqa: BLE001
        root.withdraw()
        messagebox.showerror("LoRA Classifier", f"Could not load config.toml:\n{e}")
        return
    App(root, cfg)
    root.mainloop()


if __name__ == "__main__":
    launch()

"""UI smoke tests: build the real window against the temp library and exercise the tabs."""
import time

import pytest

tk = pytest.importorskip("tkinter")


@pytest.fixture(scope="session")
def tk_root():
    """One Tcl interpreter for the whole session, like the real app. Creating and destroying a
    Tk() per test intermittently fails on Windows ("Can't find a usable init.tcl")."""
    from lora_classifier import gui  # noqa: F401 - pins the Tcl library paths first
    try:
        root = tk.Tk()
    except tk.TclError as e:
        pytest.skip(f"Tk unavailable: {e}")
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def app(cfg, library, fakes, tk_root):
    from lora_classifier import actions, gui
    actions.run(cfg)
    window = tk.Toplevel(tk_root)
    window.withdraw()
    a = gui.App(window, cfg)
    wait_loaded(a)
    yield a
    if a.worker:
        a.worker.join(timeout=30)
    pump(a, 0.2)  # drain queued UI events before the window goes away
    window.destroy()


def pump(app, seconds=0.5):
    end = time.time() + seconds
    while time.time() < end:
        app.root.update()
        time.sleep(0.02)


def wait(app, limit=10):
    end = time.time() + limit
    while app.busy() and time.time() < end:
        pump(app, 0.1)
    pump(app, 0.3)


def wait_loaded(app, limit=20):
    """Tabs load their data in the background; wait until it has arrived and been drawn."""
    end = time.time() + limit
    pump(app, 0.1)
    while (app.loading() or app._reload_again) and time.time() < end:
        pump(app, 0.1)
    pump(app, 0.3)


def lid(cfg, path):
    from lora_classifier import metadata
    return metadata.lora_id(path.relative_to(cfg.source_dir).as_posix())


def select(app, item_id):
    wait_loaded(app)
    app.review.filter.set("All")
    app.review.populate()
    app.review.tree.selection_set(item_id)
    pump(app)


def states(rev, **kw):
    return {k: str(getattr(rev, f"{k}_btn").cget("state")) for k in kw}


def test_tabs_build_and_list_records(app):
    assert [app.tabs.tab(t, "text") for t in app.tabs.tabs()] == ["Run", "Inspector", "Review"]
    assert app.inspector.count_label.cget("text") == "6 of 6 LoRAs"
    labels = {app.review.label(i) for i in app.review.items.values()}
    assert labels == {"caution", "reviewable"}
    rows = {app.review.tree.item(i, "text"): app.review.tree.item(i, "values")[0]
            for i in app.review.tree.get_children()}
    assert rows["aged.safetensors"] == "caution (held) · set aside"
    assert rows["rockwell.safetensors"] == "reviewable"


def test_filters(app):
    rev = app.review
    for f, expected in (("held", {"aged.safetensors"}), ("set aside", {"aged.safetensors"}),
                        ("reviewable", {"rockwell.safetensors"}), ("quarantined", set()), ("approved", set())):
        rev.filter.set(f)
        rev.populate()
        assert {rev.tree.item(i, "text") for i in rev.tree.get_children()} == expected, f


def test_buttons_follow_tier(app, cfg, library):
    rev = app.review
    select(app, lid(cfg, library["csam"]))  # caution, processed: reviewable and testable
    assert states(rev, approve=1, test=1, quarantine=1, release=1, block=1) == {
        "approve": "normal", "test": "normal", "quarantine": "normal", "release": "disabled", "block": "normal"}
    select(app, lid(cfg, library["aged"]))  # held: must be approved before testing
    assert states(rev, approve=1, test=1) == {"approve": "normal", "test": "disabled"}
    assert "Held" in rev.bench_note.cget("text")


def test_quarantine_then_release_through_ui(app, cfg, library, monkeypatch):
    from tkinter import simpledialog
    rev = app.review
    target = lid(cfg, library["csam"])
    select(app, target)
    monkeypatch.setattr(simpledialog, "askstring", lambda *a, **k: "reviewed: quarantine")
    rev.quarantine()
    wait(app)
    select(app, target)
    assert rev.label(rev.current) == "quarantined"
    assert states(rev, approve=1, test=1, quarantine=1, release=1) == {
        "approve": "disabled", "test": "disabled", "quarantine": "disabled", "release": "normal"}
    rev.release()
    wait(app)
    select(app, target)
    assert rev.label(rev.current) == "caution" and rev.current["status"] == "pending"


def test_ui_stays_responsive_while_data_loads(app, cfg, monkeypatch):
    """Loading review data can take seconds on a big library; it must never block the window."""
    from lora_classifier import actions
    real = actions.review_items

    def slow(c):
        time.sleep(1.5)
        return real(c)
    monkeypatch.setattr(actions, "review_items", slow)
    t = time.time()
    app.refresh_all()
    assert time.time() - t < 0.2  # returns immediately
    ticks = 0
    end = time.time() + 1.0
    while time.time() < end:  # the event loop keeps running while the loader works
        app.root.update()
        ticks += 1
        time.sleep(0.01)
    assert ticks > 30 and app.loading()
    app.refresh_all()  # a request during a load is coalesced, not run in parallel
    assert app._reload_again
    wait_loaded(app)
    assert not app.loading() and app.review.items


def test_approve_then_test_render_through_ui(app, cfg, library, fakes):
    from lora_classifier import actions
    comfy, llm = fakes
    llm.screen_verdict = lambda i: {"verdict": "possible_minor", "sexual": "none", "reason": "kid"}
    h = lid(cfg, library["child_sfw"])
    app.start_task("Approve", actions.approve, cfg, h, "ok", on_result=lambda _r: app.refresh_all())
    wait(app)
    rev = app.review
    select(app, h)
    assert str(rev.test_btn.cget("state")) == "normal"
    rev.prompt.insert("1.0", "child, kite")
    rev.count.set("1")
    rev.render_test()
    wait(app)
    assert rev.results.winfo_children(), "results should be shown"
    assert len(rev._photos) == 1  # the passing image is displayed


def test_start_button_offers_to_continue(cfg, library, fakes, tk_root):
    from lora_classifier import actions, gui
    actions.run(cfg, limit=2)  # a run that stopped part-way
    window = tk.Toplevel(tk_root)
    window.withdraw()
    a = gui.App(window, cfg)
    wait_loaded(a)
    try:
        assert a.start_btn.cget("text") == "Continue (4 left)"
        assert "2 of 6 LoRAs processed" in a.library_label.cget("text")
        assert "4 remaining" in a.library_label.cget("text")
        a.redo.set(True)  # reprocessing everything is a fresh start, not a continuation
        assert a.start_btn.cget("text") == "Start"
        a.redo.set(False)
        a.match.set("clean_style")  # the count follows the path filter
        pump(a, 0.8)
        assert "matching 'clean_style'" in a.library_label.cget("text")
    finally:
        window.destroy()

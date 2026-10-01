"""Command line front end. With no command, opens the UI."""
import argparse
import sys

from . import actions
from .comfy import ComfyError
from .config import Config
from .llm import LLMError


def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "rescreen":  # its own options (see rescreen.py)
        from .rescreen import main as rescreen_main
        return rescreen_main(sys.argv[2:])
    for stream in (sys.stdout, sys.stderr):
        if stream:
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(prog="lora_classifier", description=__doc__)
    ap.add_argument("--config", help="path to config.toml")
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("ui", help="open the UI (default when no command is given)")
    run = sub.add_parser("run", help="probe, classify, describe and illustrate LoRAs (resumable)")
    run.add_argument("--match", help="only LoRAs whose path contains this text")
    run.add_argument("--limit", type=int, help="process at most N LoRAs")
    run.add_argument("--probes", type=int, help="probe images per LoRA (default from config; 4, 8 or 16)")
    run.add_argument("--redo", action="store_true", help="reprocess LoRAs that are already finished")
    run.add_argument("--no-examples", action="store_true", help="skip example generation and evaluation")

    scan = sub.add_parser("scan", help="show detected triggers / metadata flags without rendering")
    scan.add_argument("--match")
    st = sub.add_parser("status", help="progress summary")
    st.add_argument("-v", "--verbose", action="store_true", help="also list high-risk LoRAs")
    sub.add_parser("organize", help="rebuild output/loras and descriptions from saved results")

    rt = sub.add_parser("retier", help="re-assess processed LoRAs under the current safety rules")
    rt.add_argument("--apply", action="store_true", help="apply the changes (default: dry run, report only)")
    bl = sub.add_parser("blocklist", help="find files whose hash is on the blocklist")
    bl.add_argument("--enforce", action="store_true", help="delete them (default: report only)")
    ap_ = sub.add_parser("approve", help="record a manual review approval for a LoRA id")
    ap_.add_argument("lora_id")
    ap_.add_argument("--notes", default="")
    rv = sub.add_parser("revoke", help="revoke a manual review approval")
    rv.add_argument("lora_id")
    qt = sub.add_parser("quarantine", help="manual review: quarantine a LoRA (never rendered)")
    qt.add_argument("lora_id")
    qt.add_argument("--notes", default="")
    rl = sub.add_parser("release", help="release a LoRA from quarantine (re-processed on the next run)")
    rl.add_argument("lora_id")
    rl.add_argument("--notes", default="")
    bk = sub.add_parser("block", help="blocklist a LoRA and permanently DELETE its file")
    bk.add_argument("lora_id")
    bk.add_argument("--reason", required=True)
    bk.add_argument("--yes", action="store_true", help="required: confirms the permanent deletion")

    args = ap.parse_args()
    if args.cmd in (None, "ui"):
        from .gui import launch
        launch(args.config)
        return

    cfg = Config(args.config)
    try:
        if args.cmd == "run":
            actions.run(cfg, match=args.match, limit=args.limit, probes=args.probes,
                        redo=args.redo, no_examples=args.no_examples)
        elif args.cmd == "scan":
            actions.scan(cfg, args.match)
        elif args.cmd == "status":
            actions.status(cfg, args.verbose)
        elif args.cmd == "organize":
            actions.rebuild(cfg)
        elif args.cmd == "retier":
            actions.retier(cfg, dry_run=not args.apply)
        elif args.cmd == "blocklist":
            found = actions.enforce_blocklist(cfg, dry_run=not args.enforce)
            for f in found:
                print(f"{'deleted' if f['deleted'] else 'blocked copy found'}: {f['path']}")
            print(f"{len(found)} file(s) with a blocked hash")
        elif args.cmd == "approve":
            actions.approve(cfg, args.lora_id, args.notes)
        elif args.cmd == "revoke":
            actions.revoke(cfg, args.lora_id)
        elif args.cmd == "quarantine":
            actions.quarantine(cfg, args.lora_id, args.notes)
        elif args.cmd == "release":
            actions.release(cfg, args.lora_id, args.notes)
        elif args.cmd == "block":
            if not args.yes:
                print("refusing: add --yes to confirm the file should be permanently deleted", file=sys.stderr)
                sys.exit(2)
            actions.block(cfg, args.lora_id, args.reason)
    except (ComfyError, LLMError, PermissionError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\ninterrupted; progress is saved, rerun to resume", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()

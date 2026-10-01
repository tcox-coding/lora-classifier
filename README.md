# lora-classifier

Classifies, describes and illustrates the Pony v6 LoRA library at
`...\ComfyUI-Shared\models\loras\Pony`. The source folder is **read-only** for this tool:
everything is written under `output/`.

## How it works

For each LoRA:

1. **Metadata**: reads the safetensors header (training tag frequencies, title, trained
   words, rgthree/civitai info) and detects trigger words.
2. **Probe**: local ComfyUI renders N probe images (`--probes 4|8|16`) from a generic
   template in `config.toml`, plus a few *baseline* images with the same prompt/seed but no
   LoRA, so the model can see what the LoRA actually changes.
3. **Classify + describe**: `deepseek-ai/DeepSeek-V4.1-Flash` on DeepInfra (a vision model)
   looks at the probes and metadata, picks a category path (reusing or growing the dynamic
   category tree in `output/catalog/taxonomy.json`, seeded from your hand-sorted folders),
   fills the metrics (style/position/framing/pov/subject + dynamic extras), and plans
   example images. The number of examples scales with how versatile the LoRA is.
4. **Examples**: ComfyUI renders the planned examples.
5. **Evaluate**: the model reviews the examples, scores each one, and writes the final
   description (it may revise the category).
6. **Organize**: the LoRA is hardlinked into `output/loras/Pony/<category>/`, and
   descriptions are written with the example image paths embedded.

DeepInfra doesn't host your LoRAs, so all images are rendered by your local ComfyUI. DeepSeek
only sees them.

### Civitai lookup

Each LoRA is looked up on Civitai by its file's SHA-256 (metadata only: example image sizes,
prompts, trigger words and moderation flags; images are never downloaded). This sends the
hashes to civitai.com. Hashes and responses are cached in `output/cache/`, and it can be
disabled in `config.toml` under `[civitai]`. With `probe_resolutions = true`, probes are rendered
at the resolutions the LoRA's Civitai examples use, snapped to the nearest Pony/SDXL resolution
(1024², 1152×896, 1216×832, 1344×768, 1536×640 and their portrait versions) and split in
proportion. LoRAs not on Civitai use square probes. Civitai's trigger words are added to ours,
and the model sees the example resolutions and prompts when it classifies.

## Safety tiers, manual review and the blocklist

Nothing here is configurable. See `lora_classifier/safety.py` for the exact rules.

Automatic flags never stop a LoRA from being processed; they mark it for review:

- **Reviewable:** child-related terms in a non-explicit LoRA, Civitai minor flags, or
  child-related terms in its Civitai example prompts.
- **Caution:** CSAM-coded terms (`loli`, `shota`…) in the name, captions (every tag, not just the
  most common), trigger words or Civitai prompts, or child-related terms in an explicit LoRA.
  Auto-taggers often put these on petite adults, so the LoRA is still processed.
- **Held** (a Caution LoRA): explicit under-18 ages (`9yo`, "16 years old") in its name,
  captions or trigger words. Not rendered until you approve it.
- **Quarantined:** only by manual review (Review tab). Never rendered until released. Its
  images are deleted; the LoRA file is kept unless you also Block it.

Flagged, unreviewed LoRAs render in **strict** mode: the negative prompt carries heavily
weighted CSAM terms (`(loli:1.6), (shota:1.6), (child:1.5), (underage:1.6)…`), and minor or
school tags are stripped from every prompt. As always, every image is screened in memory, and
any childlike image discards the whole batch and sets the LoRA aside as **caution** (folder
`caution loras/`) for review.

**Screening** uses two independent vision models on DeepInfra. Every image goes to the screener
(`model`, DeepSeek-V4.1-Flash), and any image it flags is re-checked by `second_opinion_model`
(Qwen3-VL-235B). The image counts as flagged only if both agree; the sexual rating is the more
severe of the two. If the second model errors or refuses, the first flag stands. This costs about
$0.0002 per flagged image. Unapproved LoRAs are screened with a strict "when in doubt, flag"
prompt; approved LoRAs with a calibrated prompt that flags minors only on clear evidence (petite
adults, anime styles, socks or school-style outfits alone aren't enough). Both verdicts are
recorded with every screened image that was flagged.

Manual review is tied to the file's SHA-256 and logged in `review/review_log.jsonl`:
- **Approve:** approvals are never revoked automatically; only you can revoke, quarantine or
  block. Approving a set-aside LoRA queues it, so Continue or Reprocess picks it up. A non-explicit
  LoRA renders in forced-SFW mode, where childlike *non-sexual* images are allowed. An explicit
  LoRA renders in reviewed-adult mode, with stronger adult cues, where any childlike image is
  discarded. Every image is still screened, and the check can't be turned off: flagged images
  are discarded one by one (never saved or shown, prompts listed under "Discarded by the
  screener"), and the rest are kept. If every probe is discarded, the LoRA is classified from its
  metadata ("needs review") and stays in the normal list.
- **Quarantine / Release**, **Revoke**, **Reprocess**, and a **test bench** (flagged images are
  never shown or saved).
- **Block** (also in the Inspector): adds the hash to `blocklist/<sha256>.json`, permanently
  deletes the .safetensors file after re-hashing it to confirm it's the same file, plus this
  tool's hardlink, images and description, and logs it in `blocklist/blocklist.log`. Before every
  run the `[blocklist] scan_dir` folder is swept, and any file with a blocked hash is deleted and
  logged as `AUTO-REJECTED`.

Records made under older rules: the Review tab re-judges them automatically. To update them on
disk, run `python -m lora_classifier retier` (a dry run), then `retier --apply`. LoRAs that
were set aside only because of their metadata are queued to be processed; LoRAs whose images
the screener flagged stay set aside.

## Adult-only enforcement (always on, not configurable)

- **Before rendering:** flagged LoRAs (see above) render in strict mode; held and quarantined LoRAs
  are not rendered.
- **Every prompt** gets `adult` added, and minor-related terms are added to the negative
  prompt. Minor-related tags are stripped from all generated or LLM-written prompts.
- **Every image** is kept in memory (ComfyUI's `SaveImageWebsocket` node, so ComfyUI writes
  nothing to disk) and screened by the vision model. Images are written only if the whole
  batch is judged clearly adult. A screening error or unparseable answer counts as a failure.
- **On any flag:** the batch is discarded, every image already saved for that LoRA is
  deleted, and the LoRA goes to `output/loras/Pony/caution loras/<original folder>/`. Its
  description starts with a brief plain-language explanation of why, followed by the technical
  detail and the exact prompts, seeds and weights that were rendered, marking which were flagged. The original file is untouched.

## Usage

Double-click **`LoRA Classifier.bat`** (or run `.venv\Scripts\python -m lora_classifier`) to
open the UI:

- **Run settings:** path filter, max LoRAs, probe images (4/8/16), generate examples on/off,
  reprocess finished LoRAs.
- **Start / Stop:** Stop finishes the current image and then halts. Nothing unscreened is
  saved, and the unfinished LoRA resumes on the next Start.
- **Inspector tab:** browse classified LoRAs in a category tree (search + status filter), read
  each one's description, metrics and scores, and flip through its example / probe / baseline
  images with a dropdown (double-click an image to open it full size). Flagged LoRAs show their
  flag; set-aside ones show why and the prompts that were rendered. Refreshes after each run.
- **Review tab:** every flagged, set-aside, approved, quarantined or blocked LoRA with the reason
  it was flagged; Approve / Revoke / Quarantine / Release / Reprocess / Block, and a test bench.
- **Preview (no rendering)** = `scan`, **Status** = `status -v`, **Rebuild output folders** =
  `organize`, plus buttons to open the catalogue and the output folder.

### Resuming after a stop or crash

Runs always continue where they left off. The Run tab shows how far the library has got
("369 of 574 LoRAs processed · 205 remaining"), and Start reads **Continue (N left)**. It
processes only LoRAs that were never started, interrupted, or failed with an error; finished and
set-aside ones are skipped unless "Reprocess" is ticked. A LoRA interrupted mid-render restarts
clean (its partial images are cleared).

Every record is written crash-safely: to a temp file, forced to disk, then swapped in. If a file
is still damaged (e.g. by a power cut), it is moved aside as `*.corrupt-<time>` and that LoRA is
simply processed again. `taxonomy.json` is rebuilt from the records. `approvals.json` and
`quarantine.json` are rebuilt from the append-only review log, so a quarantine is never lost,
and a damaged blocklist entry still blocks its hash.

The command line still works and runs the same code (`lora_classifier/actions.py`):

```
.venv\Scripts\python -m lora_classifier scan                  # metadata/trigger preview, no rendering
.venv\Scripts\python -m lora_classifier run --limit 5          # process 5 LoRAs (resumable)
.venv\Scripts\python -m lora_classifier run --match "styles/anime" --probes 8
.venv\Scripts\python -m lora_classifier run                    # everything not yet done
.venv\Scripts\python -m lora_classifier run --match foo --redo # reprocess
.venv\Scripts\python -m lora_classifier status -v
.venv\Scripts\python -m lora_classifier organize               # rebuild output folders from saved results
```

ComfyUI must be running (default `http://127.0.0.1:8188`, set in `config.toml`). The API key
is read from `.env` (`DEEPINFRA_API_KEY=...`).

Safety maintenance from the command line:

```
.venv\Scripts\python -m lora_classifier retier              # dry run: what would change under current rules
.venv\Scripts\python -m lora_classifier retier --apply      # apply it
.venv\Scripts\python -m lora_classifier quarantine <lora_id> --notes "..."
.venv\Scripts\python -m lora_classifier release <lora_id> --notes "..."
.venv\Scripts\python -m lora_classifier blocklist           # report files with a blocked hash
.venv\Scripts\python -m lora_classifier blocklist --enforce # delete them
.venv\Scripts\python -m lora_classifier block <lora_id> --reason "..." --yes
```

## Tests

```
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest
```

The suite builds a throwaway LoRA library in a temp folder with a fake ComfyUI and fake
DeepInfra. It never touches the real library or output folder, and fails if anything tries to
use the network.

## Output

```
output/
  loras/Pony/<category path>/<lora>.safetensors   hardlinks (no extra disk space)
  loras/Pony/caution loras/<original folder>/...   set aside for review (quarantined loras/ = manual)
  catalog/
    index.md / index.json                          everything, grouped by category
    taxonomy.json                                  the dynamic category tree
    descriptions/<category path>/<lora>.md|.json    detailed description + metrics + image paths
    examples/<lora id>/probe_XX.png, baseline_XX.png, example_XX.png
  state/<lora id>.json                             per-LoRA pipeline state (resume / rebuild)
```

Hardlinks share data with the original: don't edit the linked copies in place. Deleting a
link doesn't affect the original. Set `link_mode = "copy"` in `config.toml` if you'd rather
have independent copies.

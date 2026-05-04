# `data/webqa` — shard-14 toy bundle (runtime-minimal)

This tree holds **only what the `webqa_shard14_toy` walkthrough** needs at runtime.

## Original WebQA dataset

The full official release (train/val/test JSON, images, and download instructions) is published on the **[WebQA project site](https://webqna.github.io/)**. This lecture bundle is a **small slice** for local runs, not a replacement for that corpus.

## What is included

| Path | Description |
|------|-------------|
| `webqa_shard14_toy/webqa_slice/` | `webqa_questions.jsonl`, `webqa_texts.jsonl`, `webqa_images.jsonl`, `webqa_tables.jsonl` |
| `webqa_shard14_toy/manifest.json` | Toy build metadata (reference only; not a required pipeline input) |
| `WebQA_imgs_7z_chunks/imgs/all_png/shard_00014/*.png` | **Only the PNGs referenced** by the slice JSONL (60 files), not the full shard directory |

Total size is on the order of tens of MB (depends on images).

## What is **not** included (not needed for the shard-14 toy path)

- Full official `WebQA_train_val.json` / `WebQA_test.json` from the release
- VLP prediction JSON and img/txt TSV inputs for `build_shard14_toy_split.py`
- The **entire** `imgs/all_png/shard_00014` tree (only copied files that are referenced)

## Aligning with the pipeline locally

Let `ROOT` be the `colgraphrag_webqa` package root:

```bash
export WEBQA_IMGS_DIR="$ROOT/data/webqa/WebQA_imgs_7z_chunks/imgs/all_png/shard_00014"
export PATTERN_JSON_FILE_PATH="$ROOT/data/webqa/webqa_shard14_toy/webqa_slice/webqa_questions.jsonl"
```

Phase 2 `pattern.py` uses `records_for_pattern()`, which by default expects **`WebQA_train_val.json`**, but **`webqa_questions.jsonl`** loads through the same API (`*.jsonl`). With the exports above, you can drive the val slice **from this bundle alone** while keeping `WEBQA_RUN_PROFILE=val_n100`.

Image `url` fields inside the JSONL may still point at legacy dev paths (for example `/workspace/data/webqa/...`). Set **`WEBQA_IMGS_DIR`** so `inference.py` can resolve files by `image_id`.

## Publishing to Hugging Face

You can tarball this layout or push it as a Hub Dataset; students then clone and set `WEBQA_*` as above.

### Upload script

With `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) and `HF_USERNAME` in the repo `.env`:

```bash
cd /path/to/colgraphrag_webqa
pip install huggingface_hub python-dotenv   # if needed
python data/upload_webqa_to_hf.py           # default repo: {HF_USERNAME}/webqa
python data/upload_webqa_to_hf.py --dry-run
```

To change the repo id, use `HF_DATASET_REPO=you/webqa` or combine `HF_DATASET_NAME=webqa` (default) with `HF_USERNAME`. For a private dataset: `--private` or `HF_PRIVATE_DATASET=1`.

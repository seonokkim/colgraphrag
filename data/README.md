# `data/webqa` — shard-14 toy bundle (runtime-minimal)

This tree holds **only what the shard-14 toy walkthrough** needs at runtime inside the **`colgraphrag`** repo.

## Original WebQA dataset

The full official release (train/val/test JSON, images, and download instructions) is on the **[WebQA project site](https://webqna.github.io/)**. This lecture bundle is a **small slice** for local runs, not a replacement for that corpus.

## What is included

| Path | Description |
|------|-------------|
| `webqa_shard14_toy/webqa_slice/` | `webqa_questions.jsonl`, `webqa_texts.jsonl`, `webqa_images.jsonl` (**`webqa_tables.jsonl`** may appear when the toy builder emits it — not present in every minimal checkout) |
| `webqa_shard14_toy/manifest.json` | Toy build metadata (reference only; not a required pipeline input) |
| `WebQA_imgs_7z_chunks/imgs/all_png/shard_00014/*.png` | PNGs referenced by the slice JSONL (typically on the order of **60** files for this toy split), **not** the full official shard |

Total size is on the order of tens of MB (depends on images).

## What is **not** included

- Full official `WebQA_train_val.json` / `WebQA_test.json` from the release (needed only if you build the toy split from upstream assets)
- VLP prediction JSON and img/txt TSV inputs referenced by **`scripts/build_shard14_toy_split.py`**
- The **entire** upstream `imgs/all_png/shard_00014` tree (this repo keeps only PNGs referenced by the slice)

## Building or refreshing the toy split

From the **repository root** (where `scripts/` lives), with upstream WebQA + baseline files arranged as that script expects:

```bash
python scripts/build_shard14_toy_split.py
```

See the repo root **`README.md`** for full **`/workspace/data/webqa/`** layout and download pointers.

## Aligning env vars locally

Let **`ROOT`** be the repository root (folder containing **`inference.py`**):

```bash
export WEBQA_IMGS_DIR="$ROOT/data/webqa/WebQA_imgs_7z_chunks/imgs/all_png/shard_00014"
export PATTERN_JSON_FILE_PATH="$ROOT/data/webqa/webqa_shard14_toy/webqa_slice/webqa_questions.jsonl"
```

Phase 2 **`pattern.py`** can consume **`*.jsonl`** question files via the same loading path used for **`WebQA_train_val.json`**.

Image **`url`** fields in the JSONL may point at legacy machine paths (e.g. `/workspace/data/webqa/...`). Set **`WEBQA_IMGS_DIR`** (and/or align paths with this repo tree) so **`inference.py`** and the demo can resolve files by **`image_id`** / basename.

## Publishing / sharing this layout

You can tarball this subtree or publish it as a Hugging Face Dataset; recipients then unpack under **`data/webqa`** and point **`WEBQA_*`** / **`config/data.yaml`** as in the root README.

(Push automation such as **`huggingface_hub` upload helpers** belong in-repo only if checked in alongside the script; none ship in **`data/`** by default.)

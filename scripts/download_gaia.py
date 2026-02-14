#!/usr/bin/env python3
"""
Download GAIA benchmark dataset from HuggingFace.

Converts HuggingFace parquet format to per-task JSON files matching
the GAIABenchmark.load_tasks() format, and downloads attachment files
(Excel, PDF, images, audio) so the runner can pass them to the agent.

Requirements:
    pip install datasets huggingface_hub

Usage:
    # With token from env
    export HF_TOKEN=hf_xxx
    python Jotty/scripts/download_gaia.py

    # With explicit token
    python Jotty/scripts/download_gaia.py --hf-token hf_xxx

    # Custom output dir
    python Jotty/scripts/download_gaia.py --output ./data/gaia
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


GAIA_DATASET = "gaia-benchmark/GAIA"
GAIA_SUBSET = "2023_all"  # The main config name


def download_gaia(output_dir: str, hf_token: str = None, splits: list = None):
    """
    Download GAIA dataset and convert to per-task JSON files.
    Also downloads attachment files (file_path) next to each task JSON
    so the benchmark can pass them to the agent.

    Args:
        output_dir: Directory to write JSON and attachment files
        hf_token: HuggingFace API token
        splits: Which splits to download (default: ['validation', 'test'])
    """
    try:
        from datasets import load_dataset
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: 'datasets' and 'huggingface_hub' required. Install with:")
        print("  pip install datasets huggingface_hub")
        return 1

    if splits is None:
        splits = ["validation", "test"]

    output_path = Path(output_dir)
    token = hf_token or os.environ.get("HF_TOKEN")

    if not token:
        print("WARNING: No HF_TOKEN provided. GAIA may require authentication.")
        print("  Set HF_TOKEN env var or use --hf-token flag.")

    print(f"Downloading GAIA dataset ({GAIA_DATASET}) ...")
    print(f"Output: {output_path.resolve()}")

    for split_name in splits:
        print(f"\nLoading split: {split_name} ...")
        try:
            dataset = load_dataset(
                GAIA_DATASET,
                GAIA_SUBSET,
                split=split_name,
                token=token,
            )
        except Exception as e:
            print(f"  Failed to load {split_name}: {e}")
            continue

        split_dir = output_path / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        attachments_downloaded = 0
        for row in dataset:
            task_id = row.get("task_id", f"{split_name}_{count}")
            safe_id = task_id.replace("/", "_").replace("\\", "_")

            task = {
                "task_id": task_id,
                "Question": row.get("Question", ""),
                "Final answer": row.get("Final answer", ""),
                "Level": row.get("Level", 1),
                "file_name": row.get("file_name", ""),
                "Annotator Metadata": row.get("Annotator Metadata", {}),
            }

            # Download attachment if present (file_path is relative to repo root, e.g. 2023/validation/uuid.xlsx)
            file_path = (row.get("file_path") or "").strip()
            if file_path:
                try:
                    ext = Path(file_path).suffix
                    local_name = f"{safe_id}{ext}"
                    local_path = split_dir / local_name
                    downloaded = hf_hub_download(
                        repo_id=GAIA_DATASET,
                        filename=file_path,
                        repo_type="dataset",
                        token=token,
                        local_dir=str(split_dir),
                        local_dir_use_symlinks=False,
                    )
                    # Downloaded path may be under split_dir/2023/validation/...; flatten to split_dir/<task_id>.<ext>
                    downloaded_path = Path(downloaded)
                    if downloaded_path != local_path and downloaded_path.exists():
                        shutil.copy2(downloaded_path, local_path)
                    else:
                        local_path = downloaded_path
                        local_name = local_path.name
                    task["attachment_path"] = local_name
                    attachments_downloaded += 1
                except Exception as e:
                    print(f"  Warning: could not download {file_path}: {e}")

            task_file = split_dir / f"{safe_id}.json"
            task_file.write_text(json.dumps(task, indent=2, ensure_ascii=False))
            count += 1

        print(f"  Wrote {count} tasks, {attachments_downloaded} attachments to {split_dir}")

    total = sum(
        len(list((output_path / s).glob("*.json")))
        for s in splits
        if (output_path / s).exists()
    )
    print(f"\nDone! {total} total task files in {output_path.resolve()}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Download GAIA benchmark dataset from HuggingFace",
    )
    parser.add_argument(
        "--output", default="./data/gaia",
        help="Output directory (default: ./data/gaia)",
    )
    parser.add_argument(
        "--hf-token", default=None,
        help="HuggingFace API token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["validation", "test"],
        help="Splits to download (default: validation test)",
    )
    args = parser.parse_args()

    return download_gaia(
        output_dir=args.output,
        hf_token=args.hf_token,
        splits=args.splits,
    )


if __name__ == "__main__":
    sys.exit(main())

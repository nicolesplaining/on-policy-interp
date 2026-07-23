#!/usr/bin/env python3
"""Merge per-shard teacher trace outputs into single files.

  teacher_sft.shard*.jsonl -> teacher_sft.jsonl   (sorted by id)
  teacher_soft.shard*.pt    -> teacher_soft.pt     (list of per-example dicts)
"""
import argparse
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from opi.io_utils import read_jsonl, write_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="data/teacher_traces")
    args = ap.parse_args()

    hard = []
    for p in sorted(glob.glob(os.path.join(args.dir, "teacher_sft.shard*.jsonl"))):
        hard.extend(read_jsonl(p))
    hard.sort(key=lambda r: r["id"])
    hard_path = os.path.join(args.dir, "teacher_sft.jsonl")
    write_jsonl(hard_path, hard)

    soft = []
    for p in sorted(glob.glob(os.path.join(args.dir, "teacher_soft.shard*.pt"))):
        soft.extend(torch.load(p, weights_only=False))
    soft.sort(key=lambda r: r["id"])
    soft_path = os.path.join(args.dir, "teacher_soft.pt")
    torch.save(soft, soft_path)

    rhyme_rate = (sum(r.get("teacher_rhymes", False) for r in hard) / len(hard)
                  if hard else 0.0)
    print(f"Merged {len(hard)} hard traces -> {hard_path}")
    print(f"Merged {len(soft)} soft traces -> {soft_path}")
    print(f"Teacher rhyme rate: {rhyme_rate:.3f}")


if __name__ == "__main__":
    main()

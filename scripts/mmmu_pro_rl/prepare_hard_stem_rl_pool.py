#!/usr/bin/env python3
"""Build a Hard-STEM & Engineering focused external RL pool (100% zero-leakage).

Targets the exact 8 high-truncation quantitative domains identified in our forensic audit
(Architecture_and_Engineering, Electronics, Energy_and_Power, Materials,
Mechanical_Engineering, Math, Computer_Science, Finance + Physics, Chemistry, Accounting, Economics)
WITHOUT discarding long/hard 0/4 multi-step derivations, while keeping 25% coverage of the
remaining 18 subjects to prevent forgetting. Strictly excludes ext_dev image groups.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
import os
from pathlib import Path

ROOT = Path(os.environ.get("MMMU_PRO_ROOT", "."))
POOLS = ROOT / "data/prepared_pools"

HARD_QUANT_SUBJECTS = {
    "Architecture_and_Engineering",
    "Electronics",
    "Energy_and_Power",
    "Materials",
    "Mechanical_Engineering",
    "Math",
    "Computer_Science",
    "Finance",
    "Physics",
    "Chemistry",
    "Accounting",
    "Economics",
}


def main():
    rng = random.Random(20261123)
    dev_rows = [json.loads(l) for l in (POOLS / "ext_dev.jsonl").read_text().splitlines() if l.strip()]
    dev_groups = {r["image_group"] for r in dev_rows}

    # Index precomputed StructReward reference_steps from verified files
    ref_map = {}
    for fn in ("sft_verified.jsonl", "rl_verified.jsonl"):
        p = POOLS / fn
        if p.exists():
            for l in p.read_text().splitlines():
                if l.strip():
                    d = json.loads(l)
                    if d.get("reference_steps"):
                        ref_map[d["id"]] = d["reference_steps"]

    # Load ALL external candidate pools (sft_pool + rl_pool) — strictly excluding ext_dev groups
    seen_ids = set()
    hard_rows = []
    other_rows = []
    for fn in ("sft_pool.jsonl", "rl_pool.jsonl"):
        for l in (POOLS / fn).read_text().splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            if r["id"] in seen_ids or r["image_group"] in dev_groups:
                continue
            seen_ids.add(r["id"])
            if r["id"] in ref_map:
                r["reference_steps"] = ref_map[r["id"]]
            else:
                r["reference_steps"] = []
            # Strongly prefer 10-opt MC, Vision-board, and Open STEM
            if r["subject"] in HARD_QUANT_SUBJECTS:
                hard_rows.append(r)
            else:
                other_rows.append(r)

    # Balance across the 12 hard quantitative subjects (preferring standard10_aug, vision_board, open)
    by_hard = defaultdict(list)
    for r in hard_rows:
        by_hard[r["subject"]].append(r)

    selected_hard = []
    for subj, items in sorted(by_hard.items()):
        # Sort so standard10_aug / vision_board / open come first, randomized within priority
        rng.shuffle(items)
        items.sort(key=lambda x: 0 if x["view_type"] in ("standard10_aug", "vision_board", "open") else 1)
        cap = 220 if subj in {
            "Architecture_and_Engineering", "Electronics", "Energy_and_Power",
            "Materials", "Mechanical_Engineering", "Math", "Computer_Science", "Finance"
        } else 140
        selected_hard.extend(items[:cap])

    # Keep 25% coverage across remaining 18 subjects (preserving broad capability)
    by_other = defaultdict(list)
    for r in other_rows:
        by_other[r["subject"]].append(r)
    selected_other = []
    for subj, items in sorted(by_other.items()):
        rng.shuffle(items)
        items.sort(key=lambda x: 0 if x["view_type"] in ("standard10_aug", "vision_board") else 1)
        selected_other.extend(items[:35])

    combined = selected_hard + selected_other
    rng.shuffle(combined)

    out_path = POOLS / "rl_hard_stem.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in combined:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Built {out_path}: total={len(combined)} (hard_quant={len(selected_hard)}, other18={len(selected_other)})")
    print("  views:", dict(Counter(r["view_type"] for r in combined)))
    print("  with_ref_steps:", sum(bool(r["reference_steps"]) for r in combined))
    print("  top subjects:", Counter(r["subject"] for r in combined).most_common(12))


if __name__ == "__main__":
    main()

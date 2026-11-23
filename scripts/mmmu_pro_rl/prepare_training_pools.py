#!/usr/bin/env python3
"""Build strictly leakage-free, balanced English SFT + RL + ExtDev training pools
from the curated 56,860-QA external dataset (20261016 + 20261115 MMK12).

Features:
- English-only filtering (prevents Chinese rationale drift on English MMMU-Pro).
- Disjoint split by image_phash_group between SFT, RL, and External Dev.
- Balanced stratified sampling across all 30 MMMU subjects.
- 3 complementary task formats (100% external data, zero MMMU/MMMU-Pro leakage):
  1. Native Multiple-Choice (4-5 options)
  2. 10-Option Augmented External MC (same-subject external distractors, uniform A-J gold)
  3. Synthetic Vision-Board Rendered MC (question + figure + options rendered onto image)
  4. Open-Ended STEM (MMK12 + Open Main)
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import textwrap
from collections import Counter, defaultdict
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(os.environ.get("MMMU_PRO_ROOT", "."))
MAIN_PKG = ROOT / "data/mmmu_train_20261016/package"
MMK12_PKG = ROOT / "data/mmmu_expansion_mmk12_20261115"
OUT_DIR = ROOT / "data/prepared_pools"
BOARD_DIR = OUT_DIR / "vision_boards"

CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def load_font(size: int, bold: bool = False):
    idx = 1 if bold else 0
    if Path(FONT_PATHS[idx]).exists():
        return ImageFont.truetype(FONT_PATHS[idx], size)
    return ImageFont.load_default()


def is_clean_english(row: dict) -> bool:
    q = str(row.get("question") or "").strip()
    if not q or len(q) < 8 or CJK_RE.search(q):
        return False
    opts = row.get("options") or []
    for o in opts:
        if CJK_RE.search(str(o)):
            return False
    return True


def clean_option_text(opt: str) -> str:
    s = str(opt).strip()
    s = re.sub(r"^\s*(?:\([A-Z]\)|[A-Z][).:])\s*", "", s)
    return s.strip()


def normalize_mc_gold(row: dict) -> str | None:
    ans = str(row.get("answer") or "").strip()
    opts = row.get("options") or []
    if not opts or not (2 <= len(opts) <= 26):
        return None
    m = re.fullmatch(r"\s*(?:\(([A-Z])\)|([A-Z])[).:]?)\s*", ans, flags=re.I)
    if m:
        letter = (m.group(1) or m.group(2)).upper()
    else:
        m2 = re.match(r"^\s*([A-Z])[).:]\s+", ans, flags=re.I)
        if m2:
            letter = m2.group(1).upper()
        else:
            return None
    idx = ord(letter) - 65
    if 0 <= idx < len(opts):
        return letter
    return None


def resolve_images(row: dict, pkg_dir: Path) -> list[str] | None:
    # train_messages or reservoir_candidates image resolution
    paths = row.get("image_paths") or []
    if not paths or len(paths) > 3:
        return None
    resolved = []
    for p in paths:
        cand = pkg_dir / "images" / Path(p).name
        if not cand.is_file():
            return None
        resolved.append(str(cand))
    return resolved


def render_vision_board(question: str, options: list[str], img_paths: list[str], item_id: str) -> str | None:
    """Render external QA + source image(s) + options onto a clean exam-sheet image."""
    BOARD_DIR.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(f"board:{item_id}:{len(options)}".encode()).hexdigest()[:16]
    out_path = BOARD_DIR / f"{h}.png"
    if out_path.exists():
        return str(out_path)

    try:
        src_imgs = []
        for p in img_paths[:2]:
            with Image.open(p) as im:
                src_imgs.append(im.convert("RGB"))
    except Exception:
        return None

    W = 960
    pad = 28
    f_q = load_font(22, bold=True)
    f_o = load_font(20, bold=False)

    clean_q = re.sub(r"<image\s*\d*>", "[Figure]", question, flags=re.I).strip()
    q_lines = []
    for para in clean_q.splitlines():
        para = para.strip()
        if para:
            q_lines.extend(textwrap.wrap(para, width=72))
    q_lines = q_lines[:14]

    opt_lines = []
    for i, o in enumerate(options):
        prefix = f"{chr(65+i)}. "
        wrapped = textwrap.wrap(prefix + clean_option_text(o), width=74)
        opt_lines.extend(wrapped[:3])

    # Resize source images to fit nicely
    resized = []
    max_img_h = 420
    max_img_w = (W - 2 * pad - (16 if len(src_imgs) > 1 else 0)) // max(1, len(src_imgs))
    for im in src_imgs:
        w, h_im = im.size
        scale = min(max_img_w / max(1, w), max_img_h / max(1, h_im), 1.3)
        nw, nh = max(64, int(w * scale)), max(64, int(h_im * scale))
        resized.append(im.resize((nw, nh), Image.Resampling.BICUBIC))

    img_row_h = max((im.height for im in resized), default=0)
    line_h_q = 29
    line_h_o = 26
    total_h = pad + len(q_lines) * line_h_q + 18 + img_row_h + 22 + len(opt_lines) * line_h_o + pad
    if total_h > 2200:
        return None

    canvas = Image.new("RGB", (W, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    y = pad
    for line in q_lines:
        draw.text((pad, y), line, fill=(15, 15, 15), font=f_q)
        y += line_h_q
    y += 14

    # Center images horizontally
    total_w = sum(im.width for im in resized) + 16 * max(0, len(resized) - 1)
    x = (W - total_w) // 2
    for im in resized:
        canvas.paste(im, (x, y))
        x += im.width + 16
    y += img_row_h + 20

    # Light separator
    draw.line([(pad, y - 8), (W - pad, y - 8)], fill=(210, 210, 210), width=1)
    for line in opt_lines:
        draw.text((pad, y), line, fill=(25, 25, 25), font=f_o)
        y += line_h_o

    canvas.save(out_path, format="PNG")
    return str(out_path)


def make_10opt_mc(row: dict, distractor_pool: list[str], rng: random.Random) -> tuple[list[str], str] | None:
    """Expand native MC options up to 10 choices using same-subject external distractors."""
    orig_opts = [clean_option_text(o) for o in row["options"]]
    gold_letter = row["gold_letter"]
    gold_text = orig_opts[ord(gold_letter) - 65]
    seen_norm = {re.sub(r"\s+", " ", o.lower()) for o in orig_opts if o}
    expanded = list(orig_opts)

    if len(expanded) < 10 and distractor_pool:
        # Filter distractors of roughly similar character length to prevent length giveaway
        avg_len = sum(len(o) for o in orig_opts) / max(1, len(orig_opts))
        cands = [d for d in distractor_pool if 0.25 * avg_len <= len(d) <= max(40, 3.0 * avg_len)]
        if len(cands) < 12:
            cands = distractor_pool
        rng.shuffle(cands)
        for d in cands:
            dn = re.sub(r"\s+", " ", d.lower())
            if dn and dn not in seen_norm and len(d) <= 180:
                seen_norm.add(dn)
                expanded.append(d)
                if len(expanded) == 10:
                    break

    if len(expanded) < 6:
        return None
    # Shuffle uniformly so gold position is uniform over A..J
    idxs = list(range(len(expanded)))
    rng.shuffle(idxs)
    gold_old = ord(gold_letter) - 65
    new_opts = [expanded[i] for i in idxs]
    new_gold = chr(65 + idxs.index(gold_old))
    return new_opts, new_gold


def main():
    rng = random.Random(20261122)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    # Load 20261016 main package
    print("Loading 20261016 main curated dataset ...")
    with (MAIN_PKG / "reservoir_candidates.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if not is_clean_english(r):
                continue
            imgs = resolve_images(r, MAIN_PKG)
            if not imgs:
                continue
            qtype = r.get("question_type")
            if qtype == "multiple-choice":
                gl = normalize_mc_gold(r)
                if not gl:
                    continue
                opts = [clean_option_text(o) for o in r["options"]]
            elif qtype == "open":
                gl = None
                opts = []
            else:
                continue
            rat = r.get("rationale") or r.get("reasoning") or ""
            if not isinstance(rat, str) or CJK_RE.search(rat):
                rat = ""
            rows.append({
                "id": r["id"],
                "subject": r["subject"],
                "subfield": r.get("subfield"),
                "source_family": r.get("source_family"),
                "question_type": qtype,
                "question": r["question"].strip(),
                "options": opts,
                "gold_letter": gl,
                "answer": str(r["answer"]).strip(),
                "image_paths": imgs,
                "image_group": str(r.get("image_phash_group") or imgs[0]),
                "pass4_correct": (r.get("pass4") or {}).get("correct_of_four", 0),
                "native_rationale": rat.strip(),
            })

    # Load 20261115 MMK12 expansion package
    print("Loading 20261115 MMK12 curated dataset ...")
    with (MMK12_PKG / "reservoir_candidates.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if not is_clean_english(r):
                continue
            imgs = resolve_images(r, MMK12_PKG)
            if not imgs:
                continue
            ans = str(r.get("answer") or "").strip()
            if not ans or len(ans) > 120:
                continue
            rows.append({
                "id": r["id"],
                "subject": r["subject"],
                "subfield": r.get("subfield"),
                "source_family": "MMK12",
                "question_type": "open",
                "question": r["question"].strip(),
                "options": [],
                "gold_letter": None,
                "answer": ans,
                "image_paths": imgs,
                "image_group": str(r.get("image_phash_group") or imgs[0]),
                "pass4_correct": (r.get("pass4") or {}).get("correct_of_four", 0),
                "native_rationale": "",
            })

    print(f"Total clean English external rows: {len(rows)}")

    # Group by image_group so SFT / RL / ExtDev are strictly image-disjoint!
    groups_by_subj = defaultdict(list)
    by_group = defaultdict(list)
    for r in rows:
        by_group[r["image_group"]].append(r)

    for g, items in by_group.items():
        subj = items[0]["subject"]
        groups_by_subj[subj].append(items)

    sft_rows, rl_rows, dev_rows = [], [], []
    for subj in sorted(groups_by_subj):
        glist = groups_by_subj[subj]
        rng.shuffle(glist)
        # Partition groups: ~5% ExtDev (up to 10 rows), ~28% RL, ~67% SFT
        n_g = len(glist)
        n_dev = min(10, max(2, int(n_g * 0.05)))
        n_rl = min(220, max(10, int(n_g * 0.28)))
        dev_g = glist[:n_dev]
        rl_g = glist[n_dev:n_dev + n_rl]
        sft_g = glist[n_dev + n_rl:]

        def flatten_balanced(groups, cap, prefer_mixed_pass4=False):
            flat = [item for grp in groups for item in grp]
            if prefer_mixed_pass4:
                # Prioritize pass4 in {1, 2, 3} for RL variance, then 0
                mixed = [x for x in flat if x["pass4_correct"] in (1, 2, 3)]
                hard = [x for x in flat if x["pass4_correct"] == 0]
                rng.shuffle(mixed)
                rng.shuffle(hard)
                # Keep up to 60% mixed + 40% hard (or whatever is available)
                chosen = mixed[: int(cap * 0.65)] + hard[: cap - min(len(mixed), int(cap * 0.65))]
                if len(chosen) < cap:
                    rem = [x for x in flat if x["id"] not in {c["id"] for c in chosen}]
                    rng.shuffle(rem)
                    chosen.extend(rem[: cap - len(chosen)])
                return chosen[:cap]
            else:
                # For SFT, balance across sources/types within subject
                mc = [x for x in flat if x["question_type"] == "multiple-choice"]
                op = [x for x in flat if x["question_type"] == "open"]
                rng.shuffle(mc)
                rng.shuffle(op)
                if mc and op:
                    n_op = min(len(op), max(1, int(cap * 0.30)))
                    n_mc = min(len(mc), cap - n_op)
                    chosen = mc[:n_mc] + op[:n_op]
                    if len(chosen) < cap:
                        chosen += mc[n_mc : n_mc + (cap - len(chosen))]
                        chosen += op[n_op : n_op + (cap - len(chosen))]
                    return chosen[:cap]
                rng.shuffle(flat)
                return flat[:cap]

        dev_rows.extend(flatten_balanced(dev_g, 10, prefer_mixed_pass4=False))
        rl_rows.extend(flatten_balanced(rl_g, 140, prefer_mixed_pass4=True))
        sft_rows.extend(flatten_balanced(sft_g, 360, prefer_mixed_pass4=False))

    # Build per-subject distractor pools strictly within each split (zero cross-split leakage!)
    def augment_split(split_rows: list[dict], split_name: str, render_vision_frac: float = 0.25, expand10_frac: float = 0.45):
        subj_opts = defaultdict(list)
        for r in split_rows:
            if r["question_type"] == "multiple-choice":
                for o in r["options"]:
                    if 1 <= len(o) <= 150:
                        subj_opts[r["subject"]].append(o)

        out = []
        v_count = 0
        opt10_count = 0
        for r in split_rows:
            rec = dict(r)
            rec["split"] = split_name
            rec["view_type"] = "open" if r["question_type"] == "open" else "standard_mc"
            rec["display_options"] = list(r["options"])
            rec["display_gold"] = r["gold_letter"] if r["question_type"] == "multiple-choice" else r["answer"]
            rec["display_images"] = list(r["image_paths"])

            if r["question_type"] == "multiple-choice":
                # Optionally expand to 10 options (external same-subject distractors)
                if len(r["options"]) < 8 and rng.random() < expand10_frac:
                    res10 = make_10opt_mc(r, subj_opts[r["subject"]], rng)
                    if res10 is not None:
                        rec["display_options"], rec["display_gold"] = res10
                        rec["view_type"] = "standard10_aug"
                        opt10_count += 1

                # Optionally render as a Vision-board exam image!
                if rng.random() < render_vision_frac:
                    board = render_vision_board(rec["question"], rec["display_options"], rec["image_paths"], rec["id"])
                    if board:
                        rec["view_type"] = "vision_board"
                        rec["display_images"] = [board]
                        v_count += 1
            out.append(rec)
        print(f"Split {split_name}: total={len(out)}, expanded_10opt={opt10_count}, vision_board={v_count}, open={sum(x['question_type']=='open' for x in out)}")
        return out

    sft_out = augment_split(sft_rows, "sft", render_vision_frac=0.28, expand10_frac=0.50)
    rl_out = augment_split(rl_rows, "rl", render_vision_frac=0.30, expand10_frac=0.55)
    dev_out = augment_split(dev_rows, "ext_dev", render_vision_frac=0.30, expand10_frac=0.60)

    # Assert strict image_group disjointness
    g_sft = {r["image_group"] for r in sft_out}
    g_rl = {r["image_group"] for r in rl_out}
    g_dev = {r["image_group"] for r in dev_out}
    assert not (g_sft & g_rl), "Image group leak between SFT and RL!"
    assert not (g_sft & g_dev), "Image group leak between SFT and Dev!"
    assert not (g_rl & g_dev), "Image group leak between RL and Dev!"

    for name, data in [("sft_pool.jsonl", sft_out), ("rl_pool.jsonl", rl_out), ("ext_dev.jsonl", dev_out)]:
        with (OUT_DIR / name).open("w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "sft_count": len(sft_out),
        "rl_count": len(rl_out),
        "ext_dev_count": len(dev_out),
        "sft_subjects": dict(Counter(r["subject"] for r in sft_out)),
        "rl_subjects": dict(Counter(r["subject"] for r in rl_out)),
        "sft_views": dict(Counter(r["view_type"] for r in sft_out)),
        "rl_views": dict(Counter(r["view_type"] for r in rl_out)),
    }
    (OUT_DIR / "pool_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print("Saved pools to", OUT_DIR)


if __name__ == "__main__":
    main()

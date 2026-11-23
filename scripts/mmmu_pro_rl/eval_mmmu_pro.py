#!/usr/bin/env python3
"""Fast, exact MMMU-Pro (Standard10 + Vision) evaluator via SGLang (BF16).

Uses the exact official MMMU-Pro `parse_multi_choice_response` logic with
deterministic per-response fallback RNG (gold never exposed to parser).
Supports both exact official prompts and calibrated concise-CoT prompts.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import socket
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import requests

TAG_RE = re.compile(r"<image\s+(\d+)>", re.I)

DOMAIN_MAP = {
    "Art and Design": ["Art", "Art_Theory", "Design", "Music"],
    "Business": ["Accounting", "Economics", "Finance", "Manage", "Marketing"],
    "Science": ["Biology", "Chemistry", "Geography", "Math", "Physics"],
    "Health and Medicine": [
        "Basic_Medical_Science",
        "Clinical_Medicine",
        "Diagnostics_and_Laboratory_Medicine",
        "Pharmacy",
        "Public_Health",
    ],
    "Humanities and Social Science": ["History", "Literature", "Sociology", "Psychology"],
    "Tech and Engineering": [
        "Agriculture",
        "Architecture_and_Engineering",
        "Computer_Science",
        "Electronics",
        "Energy_and_Power",
        "Materials",
        "Mechanical_Engineering",
    ],
}

PROMPTS = {
    "official_cot": {
        "system": None,
        "standard10": (
            "Answer the preceding multiple choice question. "
            "The last line of your response should be of the following format: "
            "'Answer: $LETTER' (without quotes) where LETTER is one of options. "
            "Think step by step before answering."
        ),
        "vision": (
            "Write out the multiple-choice question in the image and then solve it. "
            "The last line of your response should be of the following format: "
            "'Answer: $LETTER' (without quotes) where LETTER is one of options. "
            "Think step by step before answering."
        ),
    },
    "calibrated_cot": {
        "system": None,
        "standard10": (
            "Answer the preceding multiple choice question. Think step by step concisely without "
            "repeating yourself or restarting calculations: (1) extract the key facts from the image(s), "
            "(2) derive the answer cleanly, and (3) double-check that your final option letter matches "
            "your derived value. Keep the full reasoning focused and complete. "
            "The last line of your response should be of the following format: "
            "'Answer: $LETTER' (without quotes) where LETTER is one of options."
        ),
        "vision": (
            "Write out the multiple-choice question in the image and then solve it. "
            "The last line of your response should be of the following format: "
            "'Answer: $LETTER' (without quotes) where LETTER is one of options. "
            "Think step by step before answering. "
            "Keep your reasoning concise and direct (under ~400 words) without repeating or restarting calculations."
        ),
    },
    "budget_cot": {
        "system": None,
        "standard10": (
            "Answer the preceding multiple choice question. Think step by step concisely using only the "
            "key formulas and essential steps (under ~350 words). Do not restart derivations or re-check "
            "rejected options. Once you compute the result, immediately match it to the option letter. "
            "The last line of your response should be of the following format: "
            "'Answer: $LETTER' (without quotes) where LETTER is one of options."
        ),
        "vision": (
            "Write out the multiple-choice question in the image and then solve it. "
            "The last line of your response should be of the following format: "
            "'Answer: $LETTER' (without quotes) where LETTER is one of options. "
            "Think step by step before answering. "
            "Use brief, direct steps (under ~350 words) without repeating or restarting calculations."
        ),
    },
    "deep_verify_cot": {
        "system": None,
        "standard10": (
            "Answer the preceding multiple choice question carefully and thoroughly. "
            "Think step by step: (1) inspect the image(s) carefully and read all values, labels, axes, and diagrams accurately; "
            "(2) derive the solution step by step using the exact domain principles and formulas; "
            "(3) double-check your arithmetic, units, and visual readings, and verify your result against every plausible option before deciding. "
            "The last line of your response should be of the following format: "
            "'Answer: $LETTER' (without quotes) where LETTER is one of options."
        ),
        "vision": (
            "Write out the multiple-choice question and all options shown in the image accurately, and then solve it carefully and thoroughly. "
            "Think step by step: (1) carefully inspect every diagram, table, number, and label in the image; "
            "(2) derive the solution step by step using exact domain principles; "
            "(3) double-check your calculation and verify that your chosen letter matches the exact option text in the image. "
            "The last line of your response should be of the following format: "
            "'Answer: $LETTER' (without quotes) where LETTER is one of options."
        ),
    },
}


# ---- Exact Official MMMU-Pro Parser (deterministic fallback, gold-blind) ----
class _DetRandom:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.calls = 0

    def choice(self, seq):
        self.calls += 1
        return self.rng.choice(seq)


def parse_multi_choice_response_official(response: str, all_choices: list[str], index2ans: dict[str, str], rng: _DetRandom) -> str:
    """Exact logic from MMMU-Pro official evaluate.py."""
    # First check explicit "Answer: X" line pattern as in official MMMU-Pro regex + standard fallback
    if response == "API Error" or response == "":
        return rng.choice(all_choices)

    # Check standard Answer: $LETTER format from MMMU-Pro prompt first (including markdown bold)
    for pat in [
        r"(?i:Answer)(?:\*\*)?\s*:\s*(?:\*\*\s*)?\$?\(?([A-Z])\)?\$?(?:\*\*)?(?=[.,;:\s]|$)",
        r"(?i:final\s+answer)(?:\*\*)?\s*(?i:is|:)\s*(?:\*\*\s*)?\$?\(?([A-Z])\)?\$?(?:\*\*)?(?=[.,;:\s]|$)",
    ]:
        matches = re.findall(pat, response)
        if matches:
            cand = matches[-1].upper()
            if cand in all_choices:
                return cand

    # Exact MMMU / MMMU-Pro parse_multi_choice_response body
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice}." in response:
                candidates.append(choice)

    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False

    if len(candidates) == 0:
        pred_index = rng.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    start_indexes.append(response.rfind(f"({can})"))
            else:
                for can in candidates:
                    start_indexes.append(response.rfind(f" {can} "))
        else:
            for can in candidates:
                start_indexes.append(response.lower().rfind(index2ans[can].lower()))
        pred_index = candidates[int(np.argmax(start_indexes))]
    else:
        pred_index = candidates[0]
    return pred_index


def score_item(response: str, options: list[str], gold: str, item_id: str) -> dict:
    choices = [chr(ord("A") + i) for i in range(len(options))]
    index2ans = dict(zip(choices, options))
    seed_bytes = hashlib.sha256(json.dumps(["mmmu-pro-v1", item_id, response, options], ensure_ascii=False).encode()).digest()
    rng = _DetRandom(int.from_bytes(seed_bytes, "big"))
    pred = parse_multi_choice_response_official(response, choices, index2ans, rng)
    fallback = rng.calls > 0
    correct = pred == gold
    return {
        "parsed_pred": pred,
        "correct": correct,
        "strict_correct": bool(correct and not fallback),
        "random_fallback": fallback,
    }


def encode_image_b64(path: str, max_pixels: int) -> str:
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        area = w * h
        if area > max_pixels:
            scale = math.sqrt(max_pixels / area)
            nw = max(32, int(w * scale) // 32 * 32)
            nh = max(32, int(h * scale) // 32 * 32)
            im = im.resize((nw, nh), Image.Resampling.BICUBIC)
        elif area < 4096:
            scale = math.sqrt(4096 / area)
            nw = max(32, int(w * scale) // 32 * 32)
            nh = max(32, int(h * scale) // 32 * 32)
            im = im.resize((nw, nh), Image.Resampling.BICUBIC)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def build_messages(row: dict, task: str, prompt_profile: str, max_pixels: int) -> list[dict]:
    cfg = PROMPTS[prompt_profile]
    msgs = []
    if cfg["system"]:
        msgs.append({"role": "system", "content": cfg["system"]})

    if task == "standard10":
        footer = cfg["standard10"]
        text = row["question"] + "\n" + "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(row["options"])) + "\n" + footer
        parts = []
        pos = 0
        img_keys = []
        for m in TAG_RE.finditer(text):
            if m.start() > pos:
                parts.append({"type": "text", "text": text[pos:m.start()]})
            key = str(int(m.group(1)))
            if key in row["image_paths"]:
                img_keys.append(key)
                parts.append({"type": "image", "path": row["image_paths"][key]})
            else:
                parts.append({"type": "text", "text": m.group(0)})
            pos = m.end()
        if pos < len(text):
            parts.append({"type": "text", "text": text[pos:]})
        if not img_keys and row.get("image_paths"):
            for k in sorted(row["image_paths"], key=lambda x: int(x) if str(x).isdigit() else str(x)):
                img_keys.append(k)
                parts.insert(0, {"type": "image", "path": row["image_paths"][k]})
        per_img = min(max_pixels, max(65536, (max_pixels * 2) // max(1, len(img_keys))))
        content = []
        cache = {}
        for p in parts:
            if p["type"] == "text":
                content.append(p)
            else:
                if p["path"] not in cache:
                    cache[p["path"]] = encode_image_b64(p["path"], per_img)
                content.append({"type": "image_url", "image_url": {"url": cache[p["path"]]}})
        msgs.append({"role": "user", "content": content})
    else:
        footer = cfg["vision"]
        url = encode_image_b64(row["image_path"], max_pixels)
        msgs.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": footer},
            ],
        })
    return msgs


def wait_port_free(port: int, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return True
            except OSError:
                time.sleep(0.5)
    return False


def start_sglang(
    model_path: str,
    port: int,
    log_path: Path,
    lora_path: str | None = None,
    context_length: int = 16384,
    max_running_requests: int = 384,
    cuda_graph_max_bs: int = 384,
) -> subprocess.Popen:
    if not wait_port_free(port, timeout=20.0):
        raise RuntimeError(f"Port 127.0.0.1:{port} is already in use")
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--dtype", "bfloat16",
        "--context-length", str(context_length),
        "--mem-fraction-static", "0.88",
        "--max-running-requests", str(max_running_requests),
        "--cuda-graph-max-bs-decode", str(cuda_graph_max_bs),
        "--chunked-prefill-size", "16384",
    ]
    if lora_path:
        cmd += ["--enable-lora", "--lora-paths", f"adapter={lora_path}", "--max-loras-per-batch", "1"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
    url = f"http://127.0.0.1:{port}/health"
    try:
        for _ in range(300):
            if proc.poll() is not None:
                raise RuntimeError(f"SGLang exited early ({proc.returncode}); check {log_path}")
            try:
                if requests.get(url, timeout=2).ok:
                    return proc
            except requests.RequestException:
                pass
            time.sleep(1)
        raise TimeoutError("SGLang startup timed out")
    except Exception:
        stop_proc(proc, port=port)
        raise


def stop_proc(proc: subprocess.Popen | None, port: int | None = None) -> None:
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=20)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass
        try:
            proc.wait(timeout=10)
        except Exception:
            pass
    if port is not None:
        wait_port_free(port, timeout=30.0)


def score_majority_vote(
    candidates: list[tuple[str, str | None]],
    options: list[str],
    gold: str,
    item_id: str,
) -> dict:
    """Self-consistency majority vote over K sampled responses (prefers complete non-fallback votes)."""
    if len(candidates) == 1:
        return score_item(candidates[0][0], options, gold, item_id)

    scored = []
    for k, (resp, fin) in enumerate(candidates):
        sc = score_item(resp, options, gold, f"{item_id}_s{k}")
        sc["finish_reason"] = fin
        scored.append(sc)

    # Prefer complete non-fallback votes; fall back gracefully if all truncated/fallback
    pool = [s for s in scored if not s["random_fallback"] and s["finish_reason"] != "length"]
    if not pool:
        pool = [s for s in scored if not s["random_fallback"]]
    if not pool:
        pool = scored

    counts: dict[str, int] = defaultdict(int)
    first_pos: dict[str, int] = {}
    for idx, s in enumerate(pool):
        p = s["parsed_pred"]
        counts[p] += 1
        if p not in first_pos:
            first_pos[p] = idx

    best_pred = max(counts.keys(), key=lambda p: (counts[p], -first_pos[p]))
    used_fallback = all(s["random_fallback"] for s in pool)
    correct = best_pred == gold
    return {
        "parsed_pred": best_pred,
        "correct": correct,
        "strict_correct": bool(correct and not used_fallback),
        "random_fallback": used_fallback,
        "pass_at_k_correct": any(s["correct"] for s in scored),
        "vote_counts": dict(counts),
    }


def eval_task(
    url: str,
    model_name: str,
    eval_dir: Path,
    task: str,
    prompt_profile: str,
    out_dir: Path,
    sampling: dict,
    max_pixels: int,
    workers: int,
    limit: int | None = None,
    num_samples: int = 1,
) -> dict:
    inputs = [json.loads(l) for l in (eval_dir / task / "inputs.jsonl").read_text().splitlines() if l.strip()]
    labels = {r["id"]: r for r in (json.loads(l) for l in (eval_dir / task / "labels.jsonl").read_text().splitlines() if l.strip())}
    if limit:
        inputs = inputs[:limit]

    t0 = time.monotonic()
    preds_path = out_dir / f"{task}_{prompt_profile}_predictions.jsonl"
    results = []

    def run_one(row):
        msgs = build_messages(row, task, prompt_profile, max_pixels)
        seed = int(hashlib.sha256(row["id"].encode()).hexdigest()[:7], 16)
        payload = {"model": model_name, "messages": msgs, **sampling, "seed": seed}
        if num_samples > 1:
            payload["n"] = num_samples
        req_timeout = 480 if num_samples > 1 else 180
        for attempt in range(3):
            try:
                r = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=req_timeout)
                r.raise_for_status()
                data = r.json()
                choices_api = data["choices"]
                cands = [((ch["message"].get("content") or ""), ch.get("finish_reason")) for ch in choices_api]
                usage = data.get("usage") or {}
                comp_tok = usage.get("completion_tokens", 0)
                if num_samples > 1:
                    all_trunc = all(f == "length" for _, f in cands)
                    return {
                        "id": row["id"],
                        "subject": row["subject"],
                        "response": cands[0][0],
                        "candidates": cands,
                        "finish_reason": "length" if all_trunc else "stop",
                        "completion_tokens": comp_tok,
                        "status": "completed",
                    }
                return {
                    "id": row["id"],
                    "subject": row["subject"],
                    "response": cands[0][0],
                    "finish_reason": cands[0][1],
                    "completion_tokens": comp_tok,
                    "status": "completed",
                }
            except Exception as e:
                if attempt == 2:
                    return {
                        "id": row["id"],
                        "subject": row["subject"],
                        "response": "",
                        "finish_reason": "error",
                        "completion_tokens": 0,
                        "status": f"error: {e}",
                    }
                time.sleep(2)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(run_one, row) for row in inputs]
        done = 0
        for fut in as_completed(futs):
            rec = fut.result()
            lab = labels[rec["id"]]
            if "candidates" in rec:
                sc = score_majority_vote(rec.pop("candidates"), lab["options"], lab["answer"], rec["id"])
            else:
                sc = score_item(rec["response"], lab["options"], lab["answer"], rec["id"])
            rec.update(sc)
            rec["gold"] = lab["answer"]
            results.append(rec)
            done += 1
            if done % 200 == 0 or done == len(inputs):
                acc = sum(r["correct"] for r in results) / len(results) * 100
                trunc = sum(r["finish_reason"] == "length" for r in results) / len(results) * 100
                print(f"  [{task}/{prompt_profile}] {done}/{len(inputs)} acc={acc:.2f}% trunc={trunc:.1f}% ({time.monotonic()-t0:.1f}s)", flush=True)

    results.sort(key=lambda x: x["id"])
    with preds_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(results)
    correct = sum(r["correct"] for r in results)
    strict = sum(r["strict_correct"] for r in results)
    trunc = sum(r["finish_reason"] == "length" for r in results)
    fb = sum(r["random_fallback"] for r in results)
    avg_tok = sum(r["completion_tokens"] for r in results) / max(1, n)

    by_subj = defaultdict(list)
    for r in results:
        by_subj[r["subject"]].append(r["correct"])
    subj_acc = {s: round(sum(v) / len(v) * 100, 2) for s, v in sorted(by_subj.items())}
    dom_acc = {}
    for dom, subjs in DOMAIN_MAP.items():
        vals = [c for s in subjs for c in by_subj.get(s, [])]
        if vals:
            dom_acc[dom] = round(sum(vals) / len(vals) * 100, 2)

    ret = {
        "task": task,
        "prompt_profile": prompt_profile,
        "num_samples": num_samples,
        "total": n,
        "correct": correct,
        "accuracy_pct": round(correct / max(1, n) * 100, 2),
        "strict_correct": strict,
        "strict_accuracy_pct": round(strict / max(1, n) * 100, 2),
        "truncated": trunc,
        "truncated_pct": round(trunc / max(1, n) * 100, 2),
        "random_fallback": fb,
        "avg_completion_tokens": round(avg_tok, 1),
        "elapsed_sec": round(time.monotonic() - t0, 1),
        "domain_accuracy_pct": dom_acc,
        "subject_accuracy_pct": subj_acc,
    }
    if num_samples > 1:
        pak = sum(bool(r.get("pass_at_k_correct")) for r in results)
        ret["pass_at_k_correct"] = pak
        ret["pass_at_k_accuracy_pct"] = round(pak / max(1, n) * 100, 2)
    return ret


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--eval-dir", type=Path, default=Path("eval_data/mmmu_pro"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--tasks", nargs="+", default=["standard10", "vision"], choices=["standard10", "vision"])
    p.add_argument("--profiles", nargs="+", default=["official_cot", "calibrated_cot"], choices=list(PROMPTS.keys()))
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--presence-penalty", type=float, default=1.5)
    p.add_argument("--max-pixels", type=int, default=1048576)
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--num-samples", type=int, default=1, help="Samples per question (>1 uses self-consistency majority vote)")
    p.add_argument("--max-running-requests", type=int, default=384)
    p.add_argument("--cuda-graph-max-bs", type=int, default=384)
    p.add_argument("--port", type=int, default=30100)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--reuse-url", type=str, default=None)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sampling = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "presence_penalty": args.presence_penalty,
    }
    proc = None
    try:
        if args.reuse_url:
            url = args.reuse_url.rstrip("/")
        else:
            ctx_len = max(16384, args.max_tokens + 6144)
            print(
                f"Starting SGLang server for {args.model_path} on port {args.port} "
                f"(context_length={ctx_len}, max_running_requests={args.max_running_requests}, "
                f"cuda_graph_max_bs={args.cuda_graph_max_bs}) ...",
                flush=True,
            )
            proc = start_sglang(
                args.model_path,
                args.port,
                args.out_dir / "sglang_server.log",
                context_length=ctx_len,
                max_running_requests=args.max_running_requests,
                cuda_graph_max_bs=args.cuda_graph_max_bs,
            )
            url = f"http://127.0.0.1:{args.port}"
        model_name = requests.get(f"{url}/v1/models", timeout=10).json()["data"][0]["id"]
        summary = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_path": args.model_path,
            "sampling": sampling,
            "num_samples": args.num_samples,
            "max_pixels": args.max_pixels,
            "results": {},
        }
        for prof in args.profiles:
            prof_res = {}
            for task in args.tasks:
                res = eval_task(url, model_name, args.eval_dir, task, prof, args.out_dir, sampling, args.max_pixels, args.workers, args.limit, num_samples=args.num_samples)
                prof_res[task] = res
            if "standard10" in prof_res and "vision" in prof_res:
                prof_res["overall_mean_pct"] = round(
                    (prof_res["standard10"]["accuracy_pct"] + prof_res["vision"]["accuracy_pct"]) / 2.0, 2
                )
                prof_res["overall_strict_mean_pct"] = round(
                    (prof_res["standard10"]["strict_accuracy_pct"] + prof_res["vision"]["strict_accuracy_pct"]) / 2.0, 2
                )
            summary["results"][prof] = prof_res
        (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print("\n=== SUMMARY ===")
        for prof, r in summary["results"].items():
            s10 = r.get("standard10", {})
            vis = r.get("vision", {})
            print(
                f"Profile={prof}: Standard10={s10.get('accuracy_pct')}% (strict {s10.get('strict_accuracy_pct')}%, trunc {s10.get('truncated_pct')}%) | "
                f"Vision={vis.get('accuracy_pct')}% (strict {vis.get('strict_accuracy_pct')}%, trunc {vis.get('truncated_pct')}%) | "
                f"Overall={r.get('overall_mean_pct')}%"
            )
    finally:
        stop_proc(proc, port=args.port if proc is not None else None)


if __name__ == "__main__":
    main()

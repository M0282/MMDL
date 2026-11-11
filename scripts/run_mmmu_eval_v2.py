#!/usr/bin/env python
"""
MMMU validation 평가 스크립트 v2 (E1: baseline을 '전체 응답 저장' 형태로 다시 돌리기 위한 개정판)

v1(run_mmmu_eval.py)과 달라진 점
  1) 정답/오답 상관없이 900문항 전부의 응답을 한 줄씩 JSONL로 저장한다. (v1은 오답만 저장)
  2) 문항마다 생성 토큰 수, '길이 상한에 걸려 잘렸는지', '파서가 답을 못 찾아 찍었는지(fallback)'를 기록한다.
  3) 프롬프트 방식(--prompt_mode), seed(--seed), 길이(--max_new_tokens), 해상도(--max_pixels)를 인자로 받는다.
  4) 이어서 실행(resume)은 JSONL에 이미 있는 id를 건너뛰는 방식이다. (파일이 곧 진행 기록)
  5) 채점/파서는 v1과 동일한 MMMU 공식 로직이다. 파서는 건드리지 않았다. (E0에서 별도로 수정 예정)
     그래서 원본 파서 점수가 공식 비교용으로 그대로 남는다.

조건 이름 (팀 합의: 조건 A/B/C)
  A: --prompt_mode answer_only   --max_new_tokens 512   --max_pixels 1003520   (= v1 baseline과 동일)
  B: --prompt_mode step_by_step  --max_new_tokens 2048  --max_pixels 1003520
  C: --prompt_mode step_by_step  --max_new_tokens 2048  --max_pixels 4014080   (5120*28*28, 근거 미확인 값)

사용 예 (조건 A, seed 42):
    python run_mmmu_eval_v2.py --tag A_seed42 --prompt_mode answer_only --seed 42

스모크 테스트 (과목당 2문제):
    python run_mmmu_eval_v2.py --tag smoke --limit_per_subject 2

출력 (--output_dir 아래)
    {tag}_responses.jsonl : 문항별 전체 기록 (분석의 원본 데이터)
    {tag}_summary.json    : 요약 (정확도, fallback 비율, 잘림 비율, 과목별 정확도, 소요시간)
"""
import argparse
import json
import os
import random
import re
import string
import time

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, LogitsProcessor

MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
DATA_REVISION = "98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68"

SUBJECTS = [
    "Accounting", "Agriculture", "Architecture_and_Engineering", "Art", "Art_Theory",
    "Basic_Medical_Science", "Biology", "Chemistry", "Clinical_Medicine", "Computer_Science",
    "Design", "Diagnostics_and_Laboratory_Medicine", "Economics", "Electronics",
    "Energy_and_Power", "Finance", "Geography", "History", "Literature", "Manage",
    "Marketing", "Materials", "Math", "Mechanical_Engineering", "Music", "Pharmacy",
    "Physics", "Psychology", "Public_Health", "Sociology",
]

SAMPLING_RECIPE = dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.0)
PRESENCE_PENALTY = 1.5


class PresencePenaltyLogitsProcessor(LogitsProcessor):
    """v1과 동일. EOS는 페널티 대상에서 제외."""

    def __init__(self, penalty, prompt_len, eos_token_ids):
        self.penalty = penalty
        self.prompt_len = prompt_len
        if eos_token_ids is None:
            eos_token_ids = []
        elif isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]
        self.eos_token_ids = set(eos_token_ids)

    def __call__(self, input_ids, scores):
        generated = input_ids[0, self.prompt_len:]
        if generated.numel() > 0:
            for tok in torch.unique(generated).tolist():
                if tok in self.eos_token_ids:
                    continue
                scores[0, tok] = scores[0, tok] - self.penalty
        return scores


# ---------------------------------------------------------------------------
# MMMU 공식 파싱/채점 (v1과 동일, 수정 없음)
# ---------------------------------------------------------------------------
def parse_multi_choice_response(response, all_choices, index2ans):
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
            if f" {choice} " in response:
                candidates.append(choice)
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False
    if len(candidates) == 0:
        pred_index = random.choice(all_choices)
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
        pred_index = candidates[start_indexes.index(max(start_indexes))]
    else:
        pred_index = candidates[0]
    return pred_index


def mc_parser_would_fallback(response, all_choices, index2ans):
    """공식 파서가 '후보를 하나도 못 찾아서 무작위로 찍을지'만 판정한다.
    parse_multi_choice_response의 후보 탐색 3단계를 그대로 따라 한다. (random을 소모하지 않음)"""
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "
    for choice in all_choices:
        if f"({choice})" in response:
            return False
    for choice in all_choices:
        if f" {choice} " in response:
            return False
    if len(response.split()) > 5:
        for _, ans in index2ans.items():
            if ans.lower() in response.lower():
                return False
    return True


def check_is_number(s):
    try:
        float(s.replace(",", ""))
        return True
    except ValueError:
        return False


def normalize_str(s):
    s = s.strip()
    if check_is_number(s):
        return [round(float(s.replace(",", "")), 2)]
    s = s.lower()
    if len(s) == 1:
        return [" " + s, s + " "]
    return [s]


def extract_numbers(s):
    pattern_commas = r"-?\b\d{1,3}(?:,\d{3})+\b"
    pattern_scientific = r"-?\d+(?:\.\d+)?[eE][+-]?\d+"
    pattern_simple = r"-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])"
    return (re.findall(pattern_commas, s) + re.findall(pattern_scientific, s)
            + re.findall(pattern_simple, s))


def parse_open_response(response):
    def get_key_subresponses(resp):
        resp = resp.strip().strip(".").lower()
        sub_responses = re.split(r"\.\s(?=[A-Z])|\n", resp)
        indicators = ["could be ", "so ", "is ", "thus ", "therefore ", "final ", "answer ", "result "]
        key_responses = []
        for i, sr in enumerate(sub_responses):
            local_indicators = indicators + (["="] if i == len(sub_responses) - 1 else [])
            shortest = None
            for ind in local_indicators:
                if ind in sr:
                    cand = sr.split(ind)[-1].strip()
                    if shortest is None or len(cand) < len(shortest):
                        shortest = cand
            if shortest and shortest not in [":", ",", ".", "!", "?", ";", ":", "'"]:
                key_responses.append(shortest)
        return key_responses if key_responses else [resp]

    key_responses = get_key_subresponses(response)
    pred_list = key_responses.copy()
    for r in key_responses:
        pred_list.extend(extract_numbers(r))
    tmp = []
    for p in pred_list:
        tmp.extend(normalize_str(p))
    return list(set(tmp))


def eval_multi_choice(gold_i, pred_i):
    if isinstance(gold_i, list):
        return pred_i in gold_i
    return gold_i == pred_i


def eval_open(gold_i, pred_i):
    norm_answers = []
    if isinstance(gold_i, list):
        for a in gold_i:
            norm_answers.extend(normalize_str(a))
    else:
        norm_answers = normalize_str(gold_i)
    for p in pred_i:
        if isinstance(p, str):
            for na in norm_answers:
                if isinstance(na, str) and na in p:
                    return True
        else:
            if p in norm_answers:
                return True
    return False


# ---------------------------------------------------------------------------
# 프롬프트: answer_only는 v1과 글자까지 동일, step_by_step은 diagnostic_rerun.py와 동일
# ---------------------------------------------------------------------------
def build_prompt(sample, prompt_mode):
    question = sample["question"]
    options = eval(sample["options"]) if sample.get("options") else []
    is_mc = len(options) > 0

    if is_mc:
        letters = list(string.ascii_uppercase[: len(options)])
        lines = "\n".join(f"({L}) {opt}" for L, opt in zip(letters, options))
        if prompt_mode == "answer_only":
            tail = ("Answer with the option's letter from the given choices directly, "
                    "without additional explanation.")
        else:
            tail = "Solve step by step, then answer with the option's letter from the given choices."
        return f"{question}\n{lines}\n{tail}", True, letters, dict(zip(letters, options))

    if prompt_mode == "answer_only":
        tail = "Answer the question using a single word or phrase, without additional explanation."
    else:
        tail = "Solve step by step, then give your final answer using a single word or phrase."
    return f"{question}\n{tail}", False, None, None


def gather_images(sample):
    imgs = []
    for i in range(1, 8):
        img = sample.get(f"image_{i}")
        if img is not None:
            imgs.append(img.convert("RGB"))
    return imgs


# ---------------------------------------------------------------------------
# 요약 (JSONL 기록에서 매번 다시 계산 -> resume해도 숫자가 어긋나지 않음)
# ---------------------------------------------------------------------------
def summarize(records, args, hw_spec, partial):
    n = len(records)
    n_correct = sum(r["correct"] for r in records)
    mc = [r for r in records if r["is_mc"]]
    per_subject = {}
    for s in SUBJECTS:
        rs = [r for r in records if r["subject"] == s]
        if rs:
            c = sum(r["correct"] for r in rs)
            per_subject[s] = f"{c / len(rs) * 100:.2f}% ({c}/{len(rs)})"
    return {
        "partial": partial,
        "tag": args.tag,
        "condition": {
            "prompt_mode": args.prompt_mode, "max_new_tokens": args.max_new_tokens,
            "min_pixels": args.min_pixels, "max_pixels": args.max_pixels, "seed": args.seed,
            "sampling_recipe": {**SAMPLING_RECIPE, "presence_penalty": PRESENCE_PENALTY},
        },
        "parser": "MMMU official (unmodified)",
        "hw_spec": hw_spec,
        "model_revision": args.model_revision, "data_revision": args.data_revision,
        "n": n, "n_correct": n_correct,
        "accuracy_pct": round(n_correct / n * 100, 2) if n else 0,
        "mc_n": len(mc),
        "fallback_n": sum(1 for r in mc if r["parser_fallback"]),
        "fallback_pct_of_mc": round(sum(1 for r in mc if r["parser_fallback"]) / len(mc) * 100, 2) if mc else 0,
        "truncated_n": sum(1 for r in records if r["truncated"]),
        "truncated_pct": round(sum(1 for r in records if r["truncated"]) / n * 100, 2) if n else 0,
        "mean_gen_tokens": round(sum(r["n_gen_tokens"] for r in records) / n, 1) if n else 0,
        "total_elapsed_min": round(sum(r["elapsed_sec"] for r in records) / 60, 1),
        "subject_accuracy": per_subject,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="실행 이름. 출력 파일명에 쓰임 (예: A_seed42)")
    ap.add_argument("--prompt_mode", choices=["answer_only", "step_by_step"], default="answer_only")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    ap.add_argument("--max_pixels", type=int, default=1280 * 28 * 28)
    ap.add_argument("--model_path", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--model_revision", default=MODEL_REVISION)
    ap.add_argument("--data_root", default="MMMU/MMMU")
    ap.add_argument("--data_revision", default=DATA_REVISION)
    ap.add_argument("--output_dir", default="outputs")
    ap.add_argument("--limit_per_subject", type=int, default=0, help="0이면 과목당 30문제 전체")
    ap.add_argument("--no_resume", action="store_true", help="기존 JSONL을 지우고 처음부터")
    args, _unknown = ap.parse_known_args()  # 코랩의 -f 인자 무시

    from datasets import load_dataset

    os.makedirs(args.output_dir, exist_ok=True)
    jsonl_path = os.path.join(args.output_dir, f"{args.tag}_responses.jsonl")
    summary_path = os.path.join(args.output_dir, f"{args.tag}_summary.json")

    if args.no_resume and os.path.exists(jsonl_path):
        os.remove(jsonl_path)

    records = []
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if records:
            print(f"[재시작] {jsonl_path}에서 {len(records)}문항 기록을 발견했습니다. 이어서 실행합니다.")
    done_ids = {r["id"] for r in records}

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("[1/3] 데이터 로딩...")
    subject_datasets = []
    for subj in SUBJECTS:
        ds = load_dataset(args.data_root, subj, split="validation", revision=args.data_revision)
        if args.limit_per_subject:
            ds = ds.select(range(min(args.limit_per_subject, len(ds))))
        todo = [i for i, sid in enumerate(ds["id"]) if sid not in done_ids]
        subject_datasets.append((subj, ds.select(todo)))
    total_remaining = sum(len(ds) for _, ds in subject_datasets)
    total_n = total_remaining + len(records)
    print(f"  총 {total_n}문항 중 {total_remaining}문항 남음")

    print("[2/3] 모델 로딩...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, revision=args.model_revision, dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(
        args.model_path, revision=args.model_revision,
        min_pixels=args.min_pixels, max_pixels=args.max_pixels,
    )
    hw_spec = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    eos = model.generation_config.eos_token_id
    eos_ids = set([eos] if isinstance(eos, int) else (eos or []))

    print("[3/3] 추론 + 채점 시작...")
    n_done = len(records)
    start = time.time()
    with open(jsonl_path, "a") as fout:
        for subj, ds in subject_datasets:
            for sample in ds:
                t0 = time.time()
                prompt, is_mc, letters, index2ans = build_prompt(sample, args.prompt_mode)
                images = gather_images(sample)
                content = [{"type": "image", "image": img} for img in images]
                content.append({"type": "text", "text": prompt})
                messages = [{"role": "user", "content": content}]

                inputs = processor.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt",
                ).to(model.device)
                prompt_len = inputs["input_ids"].shape[-1]
                presence = PresencePenaltyLogitsProcessor(PRESENCE_PENALTY, prompt_len, eos)
                with torch.no_grad():
                    gen_ids = model.generate(
                        **inputs, max_new_tokens=args.max_new_tokens,
                        logits_processor=[presence], **SAMPLING_RECIPE,
                    )
                gen = gen_ids[0][prompt_len:]
                n_gen = int(gen.shape[0])
                truncated = bool(n_gen >= args.max_new_tokens and int(gen[-1]) not in eos_ids)
                response = processor.decode(gen, skip_special_tokens=True).strip()

                gold = sample["answer"]
                if is_mc:
                    fallback = mc_parser_would_fallback(response, letters, index2ans)
                    pred = parse_multi_choice_response(response, letters, index2ans)
                    correct = bool(eval_multi_choice(gold, pred))
                    pred_out = pred
                else:
                    fallback = None
                    pred = parse_open_response(response)
                    correct = bool(eval_open(gold, pred))
                    pred_out = [p if isinstance(p, (str, int, float)) else str(p) for p in pred]

                rec = {
                    "id": sample["id"], "subject": subj, "is_mc": is_mc,
                    "question": sample["question"], "options": sample.get("options", "[]"),
                    "gold": gold, "response": response, "parsed_pred": pred_out,
                    "correct": correct, "parser_fallback": fallback,
                    "n_gen_tokens": n_gen, "truncated": truncated,
                    "elapsed_sec": round(time.time() - t0, 2),
                    "prompt_mode": args.prompt_mode, "seed": args.seed,
                }
                records.append(rec)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()

                n_done += 1
                acc = sum(r["correct"] for r in records) / n_done * 100
                el = time.time() - start
                eta = el / max(n_done - (total_n - total_remaining), 1) * (total_n - n_done)
                print(f"  [{n_done}/{total_n}] ({subj}) {rec['elapsed_sec']}초, {n_gen}토큰"
                      f"{' (잘림)' if truncated else ''} | 누적 {acc:.1f}% | 남은 시간 약 {eta/60:.1f}분")

                if n_done % 100 == 0:
                    with open(summary_path, "w") as f:
                        json.dump(summarize(records, args, hw_spec, True), f, ensure_ascii=False, indent=2)

    summary = summarize(records, args, hw_spec, False)
    with open(summary_path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n정확도 {summary['accuracy_pct']}% ({summary['n_correct']}/{summary['n']}) | "
          f"fallback {summary['fallback_pct_of_mc']}% | 잘림 {summary['truncated_pct']}%")
    print(f"저장: {jsonl_path}, {summary_path}")


if __name__ == "__main__":
    main()

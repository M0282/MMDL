#!/usr/bin/env python
"""
0단계 진단 실험: MMMU validation 900문항을 조건(A/B/C) × seed별로 평가한다.

조건 (MMDL_프로젝트_지식.md 5장 0단계 기준)
  A: 현재 baseline과 같은 설정  — 답만 출력 프롬프트, max_new_tokens 512,  max_pixels 1280*28*28
  B: 풀이 허용 프롬프트 + max_new_tokens 2048,                          max_pixels 1280*28*28
  C: B + 고해상도                                                      max_pixels 5120*28*28

baseline 스크립트(run_mmmu_eval.py)와 같은 것:
  - 모델/데이터 revision, sampling recipe(T=0.7, top_p=0.8, top_k=20, rep 1.0, presence 1.5)
  - MMMU 공식 파서(run_mmmu_eval.py의 함수를 그대로 import)
  - min_pixels 256*28*28
다른 것:
  - 추론 백엔드: transformers 순차 → vLLM 배치 (CoT 2048토큰 x 9회 실행을 현실적인 시간에 끝내려고)
    → 그래서 조건 A도 vLLM으로 다시 돌려서 A/B/C를 같은 백엔드끼리 비교한다.
      (A 결과가 51.89%와 조금 달라도 이상한 게 아님. 차이는 결과표에 따로 적기)
  - 정답 응답까지 전부 jsonl로 저장 (baseline은 오답만 저장했음)
  - 응답 잘림 여부(finish_reason == "length"), 생성 토큰 수 기록
  - 채점을 두 가지로 기록: 공식 파서(51.89%와 같은 기준) / 수정 파서(Chem_28 유형 수정, 메인 수치)

사용 예:
  python scripts/run_stage0_vllm.py --condition A --seed 42
  python scripts/run_stage0_vllm.py --condition B --seed 43 --limit_per_subject 2   # 스모크 테스트
"""
import argparse
import json
import os
import string
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_mmmu_eval import (  # noqa: E402  baseline과 같은 파서·상수 사용
    DATA_REVISION, MODEL_REVISION, PRESENCE_PENALTY, SAMPLING_RECIPE, SUBJECTS,
    eval_multi_choice, eval_open, gather_images, parse_multi_choice_response,
    parse_open_response,
)

CONDITIONS = {
    "A": dict(prompt_mode="answer_only", max_new_tokens=512, max_pixels=1280 * 28 * 28),
    "B": dict(prompt_mode="cot", max_new_tokens=2048, max_pixels=1280 * 28 * 28),
    "C": dict(prompt_mode="cot", max_new_tokens=2048, max_pixels=5120 * 28 * 28),
}
MIN_PIXELS = 256 * 28 * 28

# 풀이 허용 프롬프트: 공식 MMMU 질문/선택지 형식은 그대로 두고 지시문만 바꾼다.
# 마지막 줄을 "Answer: X" 형식으로 고정해서, 풀이 중간에 나온 다른 선택지 문자를
# 파서가 잘못 집는 문제(Chemistry_28 같은 사례)를 줄인다.
COT_MC = ("Think through the problem step by step. Then, on the last line, write the final "
          "answer in the form \"Answer: X\" where X is the option's letter from the given choices.")
COT_OPEN = ("Think through the problem step by step. Then, on the last line, write the final "
            "answer in the form \"Answer: ...\" using a single word, number, or phrase.")


def load_mmmu_split(data_root, subj, revision, split="validation"):
    """HF 캐시(HF_HUB_CACHE)에 있는 parquet을 직접 읽는다 (오프라인 환경용).
    캐시에 없으면 원래처럼 허브에서 받는다. 두 경우 모두 같은 revision의 같은 파일."""
    import glob
    from datasets import load_dataset
    # 1순위: 직접 구성한 폴더 (MMMU_DIR/과목/validation-*.parquet) — prepare_mmmu_local.py로 만듦
    local = os.environ.get("MMMU_DIR")
    if local:
        files = sorted(glob.glob(os.path.join(local, subj, f"{split}-*.parquet")))
        if not files:
            raise FileNotFoundError(f"{local}/{subj}/{split}-*.parquet 이 없음")
        return load_dataset("parquet", data_files={split: files}, split=split)
    cache = os.environ.get("HF_HUB_CACHE")
    if cache:
        snap = os.path.join(cache, "datasets--" + data_root.replace("/", "--"), "snapshots", revision, subj)
        files = sorted(glob.glob(os.path.join(snap, f"{split}-*.parquet")))
        if files:
            return load_dataset("parquet", data_files={split: files}, split=split)
        if os.environ.get("HF_HUB_OFFLINE") == "1":
            raise FileNotFoundError(f"캐시에 {snap}/{split}-*.parquet 이 없음")
    return load_dataset(data_root, subj, split=split, revision=revision)


def build_prompt(sample, prompt_mode):
    question = sample["question"]
    options = eval(sample["options"]) if sample.get("options") else []
    is_mc = len(options) > 0
    if is_mc:
        letters = list(string.ascii_uppercase[: len(options)])
        lines = "\n".join(f"({L}) {opt}" for L, opt in zip(letters, options))
        if prompt_mode == "answer_only":  # baseline과 글자 하나까지 동일
            instr = "Answer with the option's letter from the given choices directly, without additional explanation."
        else:
            instr = COT_MC
        return f"{question}\n{lines}\n{instr}", True, letters, dict(zip(letters, options))
    if prompt_mode == "answer_only":
        instr = "Answer the question using a single word or phrase, without additional explanation."
    else:
        instr = COT_OPEN
    return f"{question}\n{instr}", False, None, None


def final_segment(response, prompt_mode):
    """CoT 응답이면 마지막 'Answer:' 뒤만 파서에 넘긴다. 없으면 응답 전체(=공식 파서 동작)."""
    if prompt_mode != "cot":
        return response
    idx = response.lower().rfind("answer:")
    if idx == -1:
        return response
    return response[idx + len("answer:"):].strip() or response


import re  # noqa: E402

# 파서 수정 (발표 0단계 "공통 수정: 파서 수정 (Chem_28 유형)")
# Chem_28 응답 "(A) (A) = 6.87D, (B) = 4.11D" → 공식 파서는 '마지막에 나온 괄호 문자'를 골라 B로 채점.
# 수정 규칙: 응답(또는 CoT의 'Answer:' 뒷부분)이 선택지 문자로 '시작'하면 그 문자를 답으로 본다.
#   "(A) ..." / "A." / "A)" / "A:" / "A" 만 해당. "A cell is ..." 같은 문장은 해당 안 됨.
_LEAD = re.compile(r"^\s*(?:\(([A-Z])\)|([A-Z])(?:[\.\):]|\s*$))")


def parse_mc_fixed(seg, letters, index2ans):
    m = _LEAD.match(seg)
    if m:
        L = m.group(1) or m.group(2)
        if L in letters:
            return L
    return parse_multi_choice_response(seg, letters, index2ans)


def grade(response, prompt_mode, is_mc, letters, index2ans, gold):
    """반환: (수정 파서 pred, 수정 파서 정오, 공식 파서 pred, 공식 파서 정오, 파서에 넘긴 부분)
    - 공식 파서: Assignment #1(51.89%)과 같은 기준 → baseline과 직접 비교용 (응답 전체를 넘김)
    - 수정 파서: CoT면 마지막 'Answer:' 뒤만 + 선두 선택지 문자 규칙 → 0단계 '새 baseline' 기준"""
    seg = final_segment(response, prompt_mode)
    if is_mc:
        off = parse_multi_choice_response(response, letters, index2ans)
        fix = parse_mc_fixed(seg, letters, index2ans)
        return fix, bool(eval_multi_choice(gold, fix)), off, bool(eval_multi_choice(gold, off)), seg
    off = parse_open_response(response)
    fix = parse_open_response(seg)
    return fix, bool(eval_open(gold, fix)), off, bool(eval_open(gold, off)), seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=list(CONDITIONS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model_path", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--model_revision", default=MODEL_REVISION)
    ap.add_argument("--data_root", default="MMMU/MMMU")
    ap.add_argument("--data_revision", default=DATA_REVISION)
    ap.add_argument("--out_dir", default="results/stage0")
    ap.add_argument("--limit_per_subject", type=int, default=0, help="0이면 과목당 30문제 전체")
    ap.add_argument("--max_model_len", type=int, default=24576)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85,
                    help="GPU를 다른 작업과 같이 쓸 때 낮추기")
    ap.add_argument("--tag", default="", help="결과 파일명 뒤에 붙일 이름 (예: 체크포인트 이름)")
    args = ap.parse_args()

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    cond = CONDITIONS[args.condition]
    name = f"{args.condition}_seed{args.seed}" + (f"_{args.tag}" if args.tag else "")
    os.makedirs(args.out_dir, exist_ok=True)
    jsonl_path = os.path.join(args.out_dir, f"{name}.jsonl")
    meta_path = os.path.join(args.out_dir, f"{name}.meta.json")

    print(f"[조건 {args.condition}] {cond} seed={args.seed}")
    samples = []
    for subj in SUBJECTS:
        ds = load_mmmu_split(args.data_root, subj, args.data_revision)
        if args.limit_per_subject:
            ds = ds.select(range(min(args.limit_per_subject, len(ds))))
        samples += [(subj, s) for s in ds]
    print(f"  {len(samples)}문항 로드")

    mm_kwargs = {"min_pixels": MIN_PIXELS, "max_pixels": cond["max_pixels"]}
    processor = AutoProcessor.from_pretrained(args.model_path, revision=args.model_revision, **mm_kwargs)
    llm = LLM(
        model=args.model_path, revision=args.model_revision, dtype="bfloat16",
        max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": 7}, mm_processor_kwargs=mm_kwargs, seed=args.seed,
    )

    requests, metas = [], []
    for i, (subj, s) in enumerate(samples):
        prompt, is_mc, letters, index2ans = build_prompt(s, cond["prompt_mode"])
        images = gather_images(s)
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
        text = processor.apply_chat_template([{"role": "user", "content": content}],
                                             tokenize=False, add_generation_prompt=True)
        req = {"prompt": text}
        if images:
            req["multi_modal_data"] = {"image": images}
        requests.append(req)
        metas.append((subj, s, is_mc, letters, index2ans))

    # 문항마다 seed를 고정(seed*10000 + 문항 번호) → 배치 순서가 바뀌어도 같은 결과
    def sp(i):
        return SamplingParams(
            temperature=SAMPLING_RECIPE["temperature"], top_p=SAMPLING_RECIPE["top_p"],
            top_k=SAMPLING_RECIPE["top_k"], repetition_penalty=SAMPLING_RECIPE["repetition_penalty"],
            presence_penalty=PRESENCE_PENALTY, max_tokens=cond["max_new_tokens"],
            seed=args.seed * 10000 + i,
        )

    t0 = time.time()
    try:
        outputs = llm.generate(requests, [sp(i) for i in range(len(requests))])
        errors = [None] * len(requests)
    except Exception as e:  # 프롬프트가 너무 긴 문항 하나 때문에 전체가 죽는 경우 → 하나씩 다시
        print(f"[경고] 배치 생성 실패({e}). 문항별로 다시 돌립니다.")
        outputs, errors = [], []
        for i, r in enumerate(requests):
            try:
                outputs.append(llm.generate([r], [sp(i)], use_tqdm=False)[0])
                errors.append(None)
            except Exception as e2:
                outputs.append(None)
                errors.append(str(e2)[:300])
    elapsed = time.time() - t0

    n_correct = n_correct_off = 0
    with open(jsonl_path, "w") as f:
        for i, ((subj, s, is_mc, letters, index2ans), out, err) in enumerate(zip(metas, outputs, errors)):
            if out is None:
                response, n_tok, finish = "", 0, "error"
            else:
                o = out.outputs[0]
                response, n_tok, finish = o.text.strip(), len(o.token_ids), o.finish_reason
            pred, correct, pred_off, correct_off, seg = grade(
                response, cond["prompt_mode"], is_mc, letters, index2ans, s["answer"])
            n_correct += correct
            n_correct_off += correct_off
            f.write(json.dumps({
                "id": s["id"], "subject": subj, "is_mc": is_mc, "gold": s["answer"],
                "response": response, "final_segment": seg if seg != response else None,
                "parsed_pred": pred, "correct": correct,
                "parsed_pred_official": pred_off, "correct_official": correct_off,
                "n_gen_tokens": n_tok,
                "truncated": finish == "length", "finish_reason": finish, "error": err,
            }, ensure_ascii=False) + "\n")

    import torch
    meta = {
        "condition": args.condition, **cond, "min_pixels": MIN_PIXELS, "seed": args.seed,
        "model_path": args.model_path, "model_revision": args.model_revision,
        "data_revision": args.data_revision, "backend": "vllm",
        "n_samples": len(samples), "n_correct": n_correct,
        "micro_acc": n_correct / len(samples),
        "micro_acc_official_parser": n_correct_off / len(samples), "elapsed_sec": round(elapsed, 1),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        # vLLM은 gpu_memory_utilization만큼 KV캐시를 미리 잡아서 torch 수치가 의미 없음.
        # 실제 VRAM은 run_stage0.sh가 nvidia-smi로 따로 기록한다.
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "sampling": {**SAMPLING_RECIPE, "presence_penalty": PRESENCE_PENALTY},
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\n[{name}] 수정 파서 {n_correct}/{len(samples)} = {n_correct / len(samples) * 100:.2f}% | "
          f"공식 파서 {n_correct_off / len(samples) * 100:.2f}%  ({elapsed / 60:.1f}분)")
    print(f"저장: {jsonl_path}")


if __name__ == "__main__":
    main()

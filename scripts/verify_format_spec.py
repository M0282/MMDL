#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
형식 사양 재채점 검증 (E_parser, 담당: 김소리)
================================================
묻는 것: "학습 모델이 풀이 마지막 줄을 'Answer: (X)' 형식으로 끝냈다면,
          공식 파서로 채점했을 때 실제로 몇 건이 회복되나?"

- 모델을 다시 돌리지 않는다. 이미 저장된 Stage 0 조건 B 응답 파일만 쓴다.
- 공식 파서는 응답 전체에서 괄호 표기를 찾고, 여러 개면 '가장 뒤'를 고른다.
  그래서 각 응답 끝에 'Answer: (X)'를 붙이면 그 (X)가 가장 뒤 괄호가 되어 읽힌다.
  X = 그 응답에서 모델이 실제로 고른 답(저장된 parsed_pred, 형식 보정 파서 결과).
- 정확한 채점을 위해 MMMU validation의 '선택지 정보'가 필요하다 (공식 파서의
  세 번째 분기가 선택지 '텍스트'로도 매칭하기 때문). 그래서 데이터셋을 불러온다.

실행 환경: Colab (GPU 불필요, A100 아무거나). 약 5~10분.
필요한 것:
  1) 팀 레포 clone 후 PR #1(stage0-diagnostic) 브랜치의 results/stage0/B_seed*.jsonl
  2) 인터넷 (HuggingFace에서 MMMU validation을 불러옴)
"""

import argparse, ast, glob, json, os, random, re
from collections import Counter

# ----- 1. 공식 파서 (팀 레포 run_mmmu_eval.py 그대로) ---------------------------
def parse_multi_choice_response(response, all_choices, index2ans):
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "
    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:                       # (A) (B) ... 형태 먼저
        if f"({choice})" in response:
            candidates.append(choice); ans_with_brack = True
    if len(candidates) == 0:
        for choice in all_choices:                   # 괄호 없는 ' A ' 형태
            if f" {choice} " in response:
                candidates.append(choice)
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():         # 선택지 '텍스트'로 매칭
            if ans.lower() in response.lower():
                candidates.append(index); index_ans = False
    if len(candidates) == 0:
        return random.choice(all_choices)
    if len(candidates) > 1:
        start = []
        if index_ans:
            if ans_with_brack:
                for can in candidates: start.append(response.rfind(f"({can})"))
            else:
                for can in candidates: start.append(response.rfind(f" {can} "))
        else:
            for can in candidates: start.append(response.lower().rfind(index2ans[can].lower()))
        return candidates[start.index(max(start))]
    return candidates[0]

# ----- 2. MMMU validation에서 id -> (선택지 글자, index2ans) ---------------------
def build_choice_map(data_root, data_revision):
    from datasets import load_dataset
    SUBJECTS = ["Accounting","Agriculture","Architecture_and_Engineering","Art",
        "Art_Theory","Basic_Medical_Science","Biology","Chemistry","Clinical_Medicine",
        "Computer_Science","Design","Diagnostics_and_Laboratory_Medicine","Economics",
        "Electronics","Energy_and_Power","Finance","Geography","History","Literature",
        "Manage","Marketing","Materials","Math","Mechanical_Engineering","Music",
        "Pharmacy","Physics","Psychology","Public_Health","Sociology"]
    choice_map = {}
    for subj in SUBJECTS:
        ds = load_dataset(data_root, subj, split="validation", revision=data_revision)
        for ex in ds:
            opts = ex["options"]
            if isinstance(opts, str):
                try: opts = ast.literal_eval(opts)
                except Exception: opts = []
            if not opts:            # 단답형은 객관식 검증에서 제외
                continue
            letters = [chr(ord("A") + i) for i in range(len(opts))]
            index2ans = {letters[i]: str(opts[i]) for i in range(len(opts))}
            choice_map[ex["id"]] = (letters, index2ans)
    return choice_map

# ----- 3. 재채점 --------------------------------------------------------------
def rewrite_tail(response, letter):
    """사양대로: 응답 끝에 'Answer: (X)'를 붙여 마지막 괄호가 정답이 되게 한다."""
    return response.rstrip() + f"\nAnswer: ({letter})"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage0_dir", default="results/stage0", help="B_seed*.jsonl 위치")
    ap.add_argument("--data_root", default="MMMU/MMMU")
    ap.add_argument("--data_revision", default="98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68")
    ap.add_argument("--cond", default="B", help="검증할 조건 (B 또는 C)")
    args = ap.parse_args()
    random.seed(0)

    print("MMMU validation 선택지 불러오는 중...")
    cmap = build_choice_map(args.data_root, args.data_revision)
    print(f"  객관식 문항 {len(cmap)}개 로드")

    files = sorted(glob.glob(os.path.join(args.stage0_dir, f"{args.cond}_seed*.jsonl")))
    assert files, f"{args.stage0_dir}에 {args.cond}_seed*.jsonl 없음"

    print(f"\n{'seed':>6} | {'객관식':>6} | {'현재 공식':>8} | {'사양 적용후':>10} | {'회복(순증)':>10}")
    print("-" * 56)
    totals = []
    for f in files:
        rows = [json.loads(l) for l in open(f)]
        mc = [r for r in rows if r["is_mc"] and r["id"] in cmap]
        cur = sum(r["correct_official"] for r in mc)   # 현재 공식 파서 정답 수
        new = 0
        for r in mc:
            letters, index2ans = cmap[r["id"]]
            intended = r["parsed_pred"]                 # 모델이 고른 답(형식 보정 파서)
            if intended not in letters:                 # 보기 범위 밖이면 그대로 둠
                rewritten = r["response"]
            else:
                rewritten = rewrite_tail(r["response"], intended)
            pred = parse_multi_choice_response(rewritten, letters, index2ans)
            new += (pred == r["gold"])
        seed = re.search(r"seed(\d+)", f).group(1)
        print(f"{seed:>6} | {len(mc):>6} | {cur:>8} | {new:>10} | {new-cur:>+10}")
        totals.append((len(mc), cur, new))

    n = len(totals)
    anc = sum(t[0] for t in totals)/n
    acur = sum(t[1] for t in totals)/n
    anew = sum(t[2] for t in totals)/n
    print("-" * 56)
    print(f"{'평균':>6} | {anc:>6.0f} | {acur:>8.1f} | {anew:>10.1f} | {anew-acur:>+10.1f}")
    print(f"\n객관식 {anc:.0f}문항 기준, 사양 적용 시 공식 파서 정답이 평균 "
          f"{acur:.1f} → {anew:.1f}건 (순증 {anew-acur:+.1f}건).")
    print("※ 전체 900문항(단답 포함) 대비 %p로 환산해 발표 숫자와 비교하면 됨.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
MMMU test 평가 결과(results/test_eval/) 요약표 생성.

입력: results/test_eval/{run}/{cond}_seed42.jsonl + .meta.json
      (run_stage0_hf.py --split test 로 만든 결과, 문항마다 correct / correct_official 기록)
출력: results/test_eval/summary.md

정확도 표기
  - 공식 파서 = MMMU 공식 parse_multi_choice_response (Assignment #1과 같은 채점)
  - 형식 보정 파서 = 마지막 'Answer:' 뒤만 채점 (Stage 0과 같은 수정 파서)
  - macro = 과목별 정확도 평균 (Stage 0 summary와 같은 방식), micro = 전체 문항 중 정답 비율

사용:
    python scripts/summarize_test_eval.py --in_dir results/test_eval --out results/test_eval/summary.md
"""
import argparse
import glob
import json
import os
import statistics as st
from collections import defaultdict

# 표에 보일 순서와 이름
RUNS = [
    ("base", "학습 전 · 조건 A (답만)"),
    ("base_B_sub50", "학습 전 · 조건 B (풀이) — 과목당 50문항"),
    ("adapter_A", "SFT A군 (답만 학습) · 조건 A"),
    ("adapter_B", "SFT B군 (풀이 학습) · 조건 B"),
]
SUBSET_RUN = "base_B_sub50"  # 시간 제약으로 일부만 돈 run. 공정 비교용 부분집합 기준


def load_run(run_dir):
    jsonl = glob.glob(os.path.join(run_dir, "*.jsonl"))
    assert len(jsonl) == 1, f"{run_dir}에 jsonl이 1개여야 함: {jsonl}"
    rows = [json.loads(l) for l in open(jsonl[0], encoding="utf-8")]
    meta = json.load(open(jsonl[0].replace(".jsonl", ".meta.json"), encoding="utf-8"))
    return rows, meta


def stats(rows):
    by_subj = defaultdict(list)
    for r in rows:
        by_subj[r["subject"]].append(r)
    n = len(rows)

    def macro(key):
        return 100 * st.mean(sum(x[key] for x in v) / len(v) for v in by_subj.values())

    def micro(key):
        return 100 * sum(r[key] for r in rows) / n

    return {
        "n": n,
        "macro_off": macro("correct_official"),
        "micro_off": micro("correct_official"),
        "macro_fix": macro("correct"),
        "trunc": 100 * sum(bool(r.get("truncated")) for r in rows) / n,
        "tok": st.median(r["n_gen_tokens"] for r in rows),
        "by_subj_off": {s: 100 * sum(x["correct_official"] for x in v) / len(v) for s, v in by_subj.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="results/test_eval")
    ap.add_argument("--out", default="results/test_eval/summary.md")
    args = ap.parse_args()

    data = {k: load_run(os.path.join(args.in_dir, k)) for k, _ in RUNS}
    full = {k: stats(rows) for k, (rows, _) in data.items()}
    sub_ids = {r["id"] for r in data[SUBSET_RUN][0]}
    sub = {k: stats([r for r in rows if r["id"] in sub_ids]) for k, (rows, _) in data.items()}

    L = []
    L.append("# MMMU test 평가 결과\n")
    L.append("- 데이터: `MMMU/MMMU` test split (rev `98e6ac0`), 30과목 10,500문항. 정답 공개(2026-02-12)로 로컬 채점")
    L.append("- 설정: Stage 0과 동일 (transformers-batched, bs48, temperature 0.7 / top_p 0.8 / top_k 20 / presence 1.5, seed 42 1회)")
    L.append("- 조건 A = 답만 / 512토큰, 조건 B = 풀이 허용 / 2048토큰, 두 조건 모두 max_pixels 1,003,520")
    L.append("- 공식 파서 = Assignment #1과 같은 채점. 형식 보정 파서 = 마지막 `Answer:` 뒤만 채점")
    L.append("- macro = 과목별 정확도 평균, micro = 전체 문항 정답 비율")
    L.append(f"- 생성: `python scripts/summarize_test_eval.py --in_dir {args.in_dir} --out {args.out}`\n")

    L.append("## 1. 전체 결과 (각 run의 전체 문항)\n")
    L.append("| run | 문항 수 | 공식 파서 macro(%) | 공식 파서 micro(%) | 형식 보정 macro(%) | 잘림 | 생성 토큰 중앙값 |")
    L.append("|---|---|---|---|---|---|---|")
    for k, name in RUNS:
        s = full[k]
        L.append(f"| {name} | {s['n']:,} | **{s['macro_off']:.2f}** | {s['micro_off']:.2f} | {s['macro_fix']:.2f} | {s['trunc']:.1f}% | {s['tok']:.0f} |")
    L.append("")
    L.append(f"> 학습 전 조건 B는 시간 제약(약 3시간)으로 과목당 50문항, 총 {full[SUBSET_RUN]['n']:,}문항만 측정했다. 직접 비교는 아래 2번 표를 쓴다.\n")

    L.append(f"## 2. 같은 {len(sub_ids):,}문항 기준 비교 (과목당 50문항)\n")
    L.append("| run | 공식 파서(%) | 형식 보정(%) | 잘림 |")
    L.append("|---|---|---|---|")
    for k, name in RUNS:
        s = sub[k]
        L.append(f"| {name} | **{s['macro_off']:.2f}** | {s['macro_fix']:.2f} | {s['trunc']:.1f}% |")
    L.append("")
    a0, b0, a1, b1 = (sub[k]["macro_off"] for k in ("base", "base_B_sub50", "adapter_A", "adapter_B"))
    L.append("**같은 조건끼리 비교 (공식 파서)**\n")
    L.append(f"- SFT A군 vs 학습 전 조건 A: {a1:.2f} vs {a0:.2f} → **{a1 - a0:+.2f}%p**")
    L.append(f"- SFT B군 vs 학습 전 조건 B: {b1:.2f} vs {b0:.2f} → **{b1 - b0:+.2f}%p**")
    L.append(f"- SFT B군 vs SFT A군: **{b1 - a1:+.2f}%p**")
    L.append(f"- 학습 전 조건 B의 잘림 {sub['base_B_sub50']['trunc']:.1f}% → SFT B군 {sub['adapter_B']['trunc']:.1f}%: 형식 사양(`Answer: (X)`로 끝내기)은 학습됨\n")

    L.append("## 3. 과목별 정확도 (공식 파서, 각 run 전체 문항)\n")
    L.append("| 과목 | " + " | ".join(name.split(" — ")[0] for _, name in RUNS) + " |")
    L.append("|---|" + "---|" * len(RUNS))
    for subj in sorted(full["base"]["by_subj_off"]):
        L.append(f"| {subj} | " + " | ".join(f"{full[k]['by_subj_off'][subj]:.1f}" for k, _ in RUNS) + " |")
    L.append("")

    L.append("## 4. 실행 정보\n")
    L.append("| run | adapter | prompt | max_new_tokens | GPU | elapsed_sec |")
    L.append("|---|---|---|---|---|---|")
    for k, name in RUNS:
        m = data[k][1]
        L.append(f"| `{k}` | `{m.get('adapter_path')}` | {m['prompt_mode']} | {m['max_new_tokens']} | {m.get('gpu')} | {m.get('elapsed_sec')} |")
    L.append("")
    L.append("> `elapsed_sec`가 4초로 찍힌 run은 이어서 실행(resume)으로 마지막 조각만 잰 값이라 실제 소요 시간이 아니다.")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

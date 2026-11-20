#!/usr/bin/env python
"""
0단계 결과 요약 → 노션에 그대로 붙여넣을 수 있는 마크다운 생성.

  python scripts/summarize_stage0.py --in_dir results/stage0 --out results/stage0/summary.md

모든 수치는 jsonl에서 직접 계산한 실측값. 정확도는 baseline 보고서와 같은 방식
(과목별 정확도의 평균 = macro avg)으로 계산한다.
"""
import argparse
import glob
import json
import os
import statistics as st
from collections import defaultdict

FIELDS = {
    "예술·디자인": ["Art", "Art_Theory", "Design", "Music"],
    "비즈니스": ["Accounting", "Economics", "Finance", "Manage", "Marketing"],
    "과학": ["Biology", "Chemistry", "Geography", "Math", "Physics"],
    "보건·의학": ["Basic_Medical_Science", "Clinical_Medicine", "Diagnostics_and_Laboratory_Medicine",
               "Pharmacy", "Public_Health"],
    "인문·사회": ["History", "Literature", "Psychology", "Sociology"],
    "기술·공학": ["Agriculture", "Architecture_and_Engineering", "Computer_Science", "Electronics",
              "Energy_and_Power", "Materials", "Mechanical_Engineering"],
}
# 오답 분석한 약점 17과목 (baseline 정확도 50% 이하). 나머지 13과목 = 강점
WEAK17 = ["Accounting", "Architecture_and_Engineering", "Biology", "Chemistry", "Computer_Science",
          "Diagnostics_and_Laboratory_Medicine", "Economics", "Electronics", "Energy_and_Power",
          "Finance", "Geography", "Manage", "Materials", "Math", "Mechanical_Engineering", "Music",
          "Physics"]
BASELINE = 51.89  # Assignment #1, transformers 백엔드, seed 42


def load(in_dir):
    runs = defaultdict(dict)  # cond -> seed -> list[record]
    metas = defaultdict(dict)
    for p in sorted(glob.glob(os.path.join(in_dir, "*_seed*.jsonl"))):
        base = os.path.basename(p)[:-6]
        cond, seed = base.split("_seed")[0], base.split("_seed")[1].split("_")[0]
        runs[cond][seed] = [json.loads(l) for l in open(p)]
        mp = os.path.join(in_dir, base + ".meta.json")
        if os.path.exists(mp):
            metas[cond][seed] = json.load(open(mp))
    return runs, metas


def subj_acc(recs, subjects=None, key="correct"):
    by = defaultdict(list)
    for r in recs:
        if subjects is None or r["subject"] in subjects:
            by[r["subject"]].append(r.get(key, r["correct"]))
    if not by:
        return float("nan")
    return 100 * st.mean(sum(v) / len(v) for v in by.values())


def mean_sd(xs):
    if len(xs) == 1:
        return f"{xs[0]:.2f}"
    return f"{st.mean(xs):.2f} ± {st.stdev(xs):.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="results/stage0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    runs, metas = load(args.in_dir)
    conds = sorted(runs)
    if not conds:
        print(f"{args.in_dir}에 결과(jsonl)가 없음 — 실행 로그(logs/*.log)를 확인")
        return
    L = []
    L.append("# 0단계 진단 실험 결과 (MMMU validation)\n")
    L.append("- 조건 A: 답만 출력 / 512토큰 / max_pixels 1280×28² (현재 baseline 설정)")
    L.append("- 조건 B: 풀이 허용 / 2048토큰 / max_pixels 1280×28²")
    L.append("- 조건 C: 풀이 허용 / 2048토큰 / max_pixels 5120×28²")
    backends = sorted({m.get("backend", "?") for c in metas.values() for m in c.values()})
    L.append(f"- 백엔드 {', '.join(backends)}, sampling recipe는 baseline과 동일. 정확도 = 과목별 정확도 평균(macro). 표기는 seed 평균 ± 표준편차")
    L.append("- 정확도 = 수정 파서(Chem_28 유형 수정 + CoT는 마지막 'Answer:' 뒤만 채점). 공식 파서 기준 = Assignment #1과 같은 채점")
    L.append(f"- 참고: Assignment #1 baseline {BASELINE}% (transformers 백엔드, seed 42 1회)\n")

    # 1. 전체
    L.append("## 1. 조건별 전체 결과\n")
    L.append("| 조건 | seed 수 | 정확도(%) | 공식 파서 기준(%) | 약점 17과목 | 강점 13과목 | 응답 잘림 | 생성 토큰 중앙값 | 실행 시간(분, 1회 평균) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    overall = {}
    for c in conds:
        seeds = runs[c]
        accs = [subj_acc(r) for r in seeds.values()]
        accs_off = [subj_acc(r, key="correct_official") for r in seeds.values()]
        overall[c] = st.mean(accs)
        weak = [subj_acc(r, WEAK17) for r in seeds.values()]
        strong = [subj_acc(r, set(sum(FIELDS.values(), [])) - set(WEAK17)) for r in seeds.values()]
        allr = [x for r in seeds.values() for x in r]
        trunc = sum(x["truncated"] for x in allr) / len(allr) * 100
        med_tok = st.median(x["n_gen_tokens"] for x in allr)
        times = [m["elapsed_sec"] / 60 for m in metas[c].values()] if metas[c] else []
        L.append(f"| {c} | {len(seeds)} | {mean_sd(accs)} | {mean_sd(accs_off)} | {mean_sd(weak)} | {mean_sd(strong)} | "
                 f"{trunc:.1f}% | {med_tok:.0f} | {st.mean(times):.1f} |" if times else
                 f"| {c} | {len(seeds)} | {mean_sd(accs)} | {mean_sd(accs_off)} | {mean_sd(weak)} | {mean_sd(strong)} | "
                 f"{trunc:.1f}% | {med_tok:.0f} | - |")
    errs = sum(1 for c in conds for r in runs[c].values() for x in r if x.get("error"))
    if errs:
        L.append(f"\n⚠️ 생성 에러 {errs}건 (프롬프트 길이 초과 등). 오답으로 처리됨 → jsonl의 error 필드 확인")
    L.append("")

    # 2. 분야별
    L.append("## 2. 분야별 정확도 (seed 평균)\n")
    L.append("| 분야 | " + " | ".join(conds) + " |")
    L.append("|---|" + "---|" * len(conds))
    for fname, subs in FIELDS.items():
        row = [f"{st.mean(subj_acc(r, subs) for r in runs[c].values()):.1f}" for c in conds]
        L.append(f"| {fname} | " + " | ".join(row) + " |")
    L.append("")

    # 3. 과목별
    L.append("## 3. 과목별 정확도 (seed 평균, 약점 과목 ★)\n")
    L.append("| 과목 | " + " | ".join(conds) + (" | B−A | C−B |" if {"A", "B", "C"} <= set(conds) else " |"))
    L.append("|---|" + "---|" * (len(conds) + (2 if {"A", "B", "C"} <= set(conds) else 0)))
    for s in sum(FIELDS.values(), []):
        vals = {c: st.mean(subj_acc(r, [s]) for r in runs[c].values()) for c in conds}
        row = [f"{vals[c]:.1f}" for c in conds]
        if {"A", "B", "C"} <= set(conds):
            row += [f"{vals['B'] - vals['A']:+.1f}", f"{vals['C'] - vals['B']:+.1f}"]
        L.append(f"| {'★ ' if s in WEAK17 else ''}{s} | " + " | ".join(row) + " |")
    L.append("")

    # 4. 문항 단위 비교
    def per_q(c):  # 문항별 seed 정답 비율
        d = defaultdict(list)
        for r in runs[c].values():
            for x in r:
                d[x["id"]].append(x["correct"])
        return {k: sum(v) / len(v) for k, v in d.items()}

    pairs = [(a, b) for a, b in [("A", "B"), ("B", "C")] if a in runs and b in runs]
    if pairs:
        L.append("## 4. 문항 단위 변화 (seed 과반 정답 기준)\n")
        L.append("| 비교 | 틀림→맞음 | 맞음→틀림 | 순변화 |")
        L.append("|---|---|---|---|")
        details = []
        for a, b in pairs:
            qa, qb = per_q(a), per_q(b)
            ids = qa.keys() & qb.keys()
            fixed = sorted(i for i in ids if qa[i] <= 0.5 and qb[i] > 0.5)
            broke = sorted(i for i in ids if qa[i] > 0.5 and qb[i] <= 0.5)
            L.append(f"| {a}→{b} | {len(fixed)} | {len(broke)} | {len(fixed) - len(broke):+d} |")
            details.append((a, b, fixed, broke))
        L.append("")
        for a, b, fixed, broke in details:
            L.append(f"<details><summary>{a}→{b} 문항 목록</summary>\n")
            L.append(f"- 틀림→맞음: {', '.join(fixed) or '없음'}")
            L.append(f"- 맞음→틀림: {', '.join(broke) or '없음'}\n</details>\n")

    # 5. 핵심 사례 추적
    watch = ["validation_Chemistry_13", "validation_Chemistry_8", "validation_Chemistry_28",
             "validation_Math_13", "validation_Math_15", "validation_Electronics_2",
             "validation_Geography_16"]
    L.append("## 5. 오답 분석 핵심 사례 추적 (seed별 정답 수)\n")
    L.append("| 문항 | " + " | ".join(conds) + " |")
    L.append("|---|" + "---|" * len(conds))
    for w in watch:
        row = []
        for c in conds:
            hits = [x["correct"] for r in runs[c].values() for x in r if x["id"] == w]
            row.append(f"{sum(hits)}/{len(hits)}" if hits else "-")
        L.append(f"| {w.replace('validation_', '')} | " + " | ".join(row) + " |")
    L.append("")

    text = "\n".join(L)
    if args.out:
        open(args.out, "w").write(text)
        print(f"저장: {args.out}")
    print(text)


if __name__ == "__main__":
    main()

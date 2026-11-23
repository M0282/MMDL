# MMMU test 평가 결과

- 데이터: `MMMU/MMMU` test split (rev `98e6ac0`), 30과목 10,500문항. 정답 공개(2026-02-12)로 로컬 채점
- 설정: Stage 0과 동일 (transformers-batched, bs48, temperature 0.7 / top_p 0.8 / top_k 20 / presence 1.5, seed 42 1회)
- 조건 A = 답만 / 512토큰, 조건 B = 풀이 허용 / 2048토큰, 두 조건 모두 max_pixels 1,003,520
- 공식 파서 = Assignment #1과 같은 채점. 형식 보정 파서 = 마지막 `Answer:` 뒤만 채점
- macro = 과목별 정확도 평균, micro = 전체 문항 정답 비율
- 생성: `python scripts/summarize_test_eval.py --in_dir results/test_eval --out results/test_eval/summary.md`

## 1. 전체 결과 (각 run의 전체 문항)

| run | 문항 수 | 공식 파서 macro(%) | 공식 파서 micro(%) | 형식 보정 macro(%) | 잘림 | 생성 토큰 중앙값 |
|---|---|---|---|---|---|---|
| 학습 전 · 조건 A (답만) | 10,500 | **51.02** | 48.50 | 51.04 | 0.8% | 5 |
| 학습 전 · 조건 B (풀이) — 과목당 50문항 | 1,500 | **47.73** | 47.73 | 58.20 | 30.3% | 615 |
| SFT A군 (답만 학습) · 조건 A | 10,500 | **42.46** | 40.68 | 42.47 | 0.0% | 1 |
| SFT B군 (풀이 학습) · 조건 B | 10,500 | **48.70** | 47.27 | 49.09 | 0.1% | 73 |

> 학습 전 조건 B는 시간 제약(약 3시간)으로 과목당 50문항, 총 1,500문항만 측정했다. 직접 비교는 아래 2번 표를 쓴다.

## 2. 같은 1,500문항 기준 비교 (과목당 50문항)

| run | 공식 파서(%) | 형식 보정(%) | 잘림 |
|---|---|---|---|
| 학습 전 · 조건 A (답만) | **51.40** | 51.47 | 0.8% |
| 학습 전 · 조건 B (풀이) — 과목당 50문항 | **47.73** | 58.20 | 30.3% |
| SFT A군 (답만 학습) · 조건 A | **42.73** | 42.80 | 0.0% |
| SFT B군 (풀이 학습) · 조건 B | **47.00** | 47.00 | 0.0% |

**같은 조건끼리 비교 (공식 파서)**

- SFT A군 vs 학습 전 조건 A: 42.73 vs 51.40 → **-8.67%p**
- SFT B군 vs 학습 전 조건 B: 47.00 vs 47.73 → **-0.73%p**
- SFT B군 vs SFT A군: **+4.27%p**
- 학습 전 조건 B의 잘림 30.3% → SFT B군 0.0%: 형식 사양(`Answer: (X)`로 끝내기)은 학습됨

## 3. 과목별 정확도 (공식 파서, 각 run 전체 문항)

| 과목 | 학습 전 · 조건 A (답만) | 학습 전 · 조건 B (풀이) | SFT A군 (답만 학습) · 조건 A | SFT B군 (풀이 학습) · 조건 B |
|---|---|---|---|---|
| Accounting | 38.7 | 56.0 | 36.8 | 58.2 |
| Agriculture | 49.8 | 42.0 | 36.9 | 38.7 |
| Architecture_and_Engineering | 32.7 | 40.0 | 29.8 | 33.4 |
| Art | 71.0 | 44.0 | 57.6 | 61.0 |
| Art_Theory | 70.2 | 38.0 | 57.3 | 59.9 |
| Basic_Medical_Science | 61.7 | 48.0 | 50.6 | 57.1 |
| Biology | 46.7 | 52.0 | 38.3 | 45.2 |
| Chemistry | 37.6 | 52.0 | 31.2 | 37.6 |
| Clinical_Medicine | 55.1 | 38.0 | 42.2 | 43.4 |
| Computer_Science | 51.2 | 46.0 | 41.2 | 48.8 |
| Design | 74.6 | 54.0 | 65.7 | 62.1 |
| Diagnostics_and_Laboratory_Medicine | 43.2 | 30.0 | 34.0 | 35.8 |
| Economics | 52.4 | 54.0 | 42.3 | 50.6 |
| Electronics | 32.4 | 28.0 | 25.4 | 26.6 |
| Energy_and_Power | 41.0 | 40.0 | 41.4 | 43.3 |
| Finance | 33.8 | 52.0 | 30.7 | 45.4 |
| Geography | 45.8 | 36.0 | 40.9 | 41.9 |
| History | 74.1 | 56.0 | 62.2 | 65.1 |
| Literature | 83.0 | 56.0 | 70.5 | 75.0 |
| Manage | 37.1 | 44.0 | 30.6 | 44.1 |
| Marketing | 55.8 | 70.0 | 38.1 | 51.9 |
| Materials | 38.2 | 32.0 | 35.6 | 36.9 |
| Math | 40.8 | 50.0 | 34.1 | 42.6 |
| Mechanical_Engineering | 46.6 | 40.0 | 38.9 | 43.6 |
| Music | 37.7 | 20.0 | 29.9 | 35.3 |
| Pharmacy | 59.5 | 72.0 | 47.0 | 60.0 |
| Physics | 42.6 | 64.0 | 35.3 | 49.5 |
| Psychology | 67.5 | 70.0 | 51.5 | 55.4 |
| Public_Health | 44.8 | 60.0 | 37.7 | 53.8 |
| Sociology | 64.7 | 48.0 | 59.9 | 58.7 |

## 4. 실행 정보

| run | adapter | prompt | max_new_tokens | GPU | elapsed_sec |
|---|---|---|---|---|---|
| `base` | `None` | answer_only | 512 | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition | 4.0 |
| `base_B_sub50` | `None` | cot | 2048 | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition | 10940.5 |
| `adapter_A` | `adapters/adapter_A` | answer_only | 512 | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition | 4.0 |
| `adapter_B` | `adapters/adapter_B` | cot | 2048 | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition | 4.0 |

> `elapsed_sec`가 4초로 찍힌 run은 이어서 실행(resume)으로 마지막 조각만 잰 값이라 실제 소요 시간이 아니다.

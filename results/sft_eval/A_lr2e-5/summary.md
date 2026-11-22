# 0단계 진단 실험 결과 (MMMU validation)

- 조건 A: 답만 출력 / 512토큰 / max_pixels 1280×28² (현재 baseline 설정)
- 조건 B: 풀이 허용 / 2048토큰 / max_pixels 1280×28²
- 조건 C: 풀이 허용 / 2048토큰 / max_pixels 5120×28²
- 백엔드 transformers-batched, sampling recipe는 baseline과 동일. 정확도 = 과목별 정확도 평균(macro). 표기는 seed 평균 ± 표준편차
- 정확도 = 수정 파서(Chem_28 유형 수정 + CoT는 마지막 'Answer:' 뒤만 채점). 공식 파서 기준 = Assignment #1과 같은 채점
- 참고: Assignment #1 baseline 51.89% (transformers 백엔드, seed 42 1회)

## 1. 조건별 전체 결과

| 조건 | seed 수 | 정확도(%) | 공식 파서 기준(%) | 약점 17과목 | 강점 13과목 | 응답 잘림 | 생성 토큰 중앙값 | 실행 시간(분, 1회 평균) |
|---|---|---|---|---|---|---|---|---|
| A | 3 | 47.56 ± 1.64 | 47.56 ± 1.64 | 37.12 ± 1.96 | 61.20 ± 2.24 | 0.1% | 1 | 2.3 |

## 2. 분야별 정확도 (seed 평균)

| 분야 | A |
|---|---|
| 예술·디자인 | 62.2 |
| 비즈니스 | 40.0 |
| 과학 | 38.4 |
| 보건·의학 | 49.3 |
| 인문·사회 | 68.1 |
| 기술·공학 | 38.1 |

## 3. 과목별 정확도 (seed 평균, 약점 과목 ★)

| 과목 | A |
|---|---|
| Art | 57.8 |
| Art_Theory | 76.7 |
| Design | 75.6 |
| ★ Music | 38.9 |
| ★ Accounting | 30.0 |
| ★ Economics | 48.9 |
| ★ Finance | 31.1 |
| ★ Manage | 37.8 |
| Marketing | 52.2 |
| ★ Biology | 43.3 |
| ★ Chemistry | 30.0 |
| ★ Geography | 32.2 |
| ★ Math | 50.0 |
| ★ Physics | 36.7 |
| Basic_Medical_Science | 56.7 |
| Clinical_Medicine | 56.7 |
| ★ Diagnostics_and_Laboratory_Medicine | 31.1 |
| Pharmacy | 56.7 |
| Public_Health | 45.6 |
| History | 72.2 |
| Literature | 75.6 |
| Psychology | 65.6 |
| Sociology | 58.9 |
| Agriculture | 45.6 |
| ★ Architecture_and_Engineering | 30.0 |
| ★ Computer_Science | 37.8 |
| ★ Electronics | 37.8 |
| ★ Energy_and_Power | 41.1 |
| ★ Materials | 32.2 |
| ★ Mechanical_Engineering | 42.2 |

## 5. 오답 분석 핵심 사례 추적 (seed별 정답 수)

| 문항 | A |
|---|---|
| Chemistry_13 | 0/3 |
| Chemistry_8 | 1/3 |
| Chemistry_28 | 3/3 |
| Math_13 | 1/3 |
| Math_15 | 0/3 |
| Electronics_2 | 3/3 |
| Geography_16 | 1/3 |

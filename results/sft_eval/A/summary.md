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
| A | 3 | 44.19 ± 0.61 | 44.19 ± 0.61 | 34.90 ± 1.09 | 56.32 ± 2.63 | 0.0% | 1 | 1.1 |

## 2. 분야별 정확도 (seed 평균)

| 분야 | A |
|---|---|
| 예술·디자인 | 57.2 |
| 비즈니스 | 38.7 |
| 과학 | 37.8 |
| 보건·의학 | 44.0 |
| 인문·사회 | 62.5 |
| 기술·공학 | 34.9 |

## 3. 과목별 정확도 (seed 평균, 약점 과목 ★)

| 과목 | A |
|---|---|
| Art | 61.1 |
| Art_Theory | 66.7 |
| Design | 70.0 |
| ★ Music | 31.1 |
| ★ Accounting | 28.9 |
| ★ Economics | 45.6 |
| ★ Finance | 31.1 |
| ★ Manage | 35.6 |
| Marketing | 52.2 |
| ★ Biology | 37.8 |
| ★ Chemistry | 28.9 |
| ★ Geography | 40.0 |
| ★ Math | 47.8 |
| ★ Physics | 34.4 |
| Basic_Medical_Science | 57.8 |
| Clinical_Medicine | 48.9 |
| ★ Diagnostics_and_Laboratory_Medicine | 37.8 |
| Pharmacy | 38.9 |
| Public_Health | 36.7 |
| History | 67.8 |
| Literature | 75.6 |
| Psychology | 53.3 |
| Sociology | 53.3 |
| Agriculture | 50.0 |
| ★ Architecture_and_Engineering | 23.3 |
| ★ Computer_Science | 42.2 |
| ★ Electronics | 24.4 |
| ★ Energy_and_Power | 45.6 |
| ★ Materials | 30.0 |
| ★ Mechanical_Engineering | 28.9 |

## 5. 오답 분석 핵심 사례 추적 (seed별 정답 수)

| 문항 | A |
|---|---|
| Chemistry_13 | 0/3 |
| Chemistry_8 | 2/3 |
| Chemistry_28 | 2/3 |
| Math_13 | 1/3 |
| Math_15 | 0/3 |
| Electronics_2 | 0/3 |
| Geography_16 | 1/3 |

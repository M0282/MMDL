# MMMU / MMMU-Pro 일반화 학습 데이터셋 구축 고려사항 및 설계 보고서

> **목적**: 고정 베이스 모델(`Qwen/Qwen3-VL-4B-Instruct`, rev `ebb281ec70b05090aa6165b016eac8ec08e71b17`)의 **MMMU / MMMU-Pro** 성능을 **벤치마크 데이터 유출(leakage) 없이** 끌어올리기 위해 구축한 56,860건 규모 멀티모달 학습 데이터셋의 수집 원칙, 필터링 게이트, 분포 현황, 그리고 후속 학습(SFT + RL) 설계 시 반영해야 할 핵심 시사점을 정리한다.

---

## 1. 핵심 설계 원칙

1. **엄격한 Zero-Leakage (벤치마크 완전 격리)**
   - MMMU(`dev`, `validation`, `test` — 2026년 2월 test 정답 공개 포함) 및 MMMU-Pro(`Standard-4`, `Standard-10`, `Vision`)의 문제·선택지·이미지·정답은 학습 데이터, 과목 분류기 학습, 리워드 설계 어디에도 일절 사용하지 않았다.
   - 벤치마크에서 가져다 쓴 정보는 오직 **30개 과목명과 183개 세부 분야(subfield) taxonomy 정의**뿐이다.
2. **원본 공개 QA 보존 (No Synthetic Authoring)**
   - 새로운 문항을 임의로 창작하거나 정답을 재작성하지 않고, 학술·공개 데이터셋의 원본 질문·정답·이미지 바이트를 그대로 보존했다.
   - 문항 표현만 바꾼 변형(title/no-title, instruction 변형)은 고유 문항 수에 산입하지 않았다.
3. **Fail-Closed 파이프라인 (`training/build_dataset.py`)**
   - 모든 품질·오염·라이선스·시각 의존성·난이도 게이트를 통과하고 증빙 메타데이터(`reservoir_candidates.jsonl`)가 완전히 기록된 행만 최종 패키지에 편입했다. 증빙이 하나라도 누락되면 탈락(fail-closed) 처리했다.
4. **완전한 재현성 및 무결성 추적**
   - 모든 원천 데이터셋의 HuggingFace/GitHub 커밋 리비전, split, 라이선스를 기록(`SOURCES.csv`)하고, 모든 이미지 및 JSONL 파일에 `SHA256SUMS` 무결성 체크섬을 부여했다.

---

## 2. 데이터 구축 기준의 진화

| 단계 | 시점 | 목표 기준 | 핵심 변화 |
|---|---|---|---|
| **1단계 (초기 레시피)** | 2026-09 | MMMU 30과목 × 과목당 ≥ 5,000 QA (총 150K) | BMMR + FineVision 백본 + 과목별 전용 보강원 라우팅 |
| **2단계 (수량 현실화)** | 2026-10-01 | 30과목 × 과목당 ≥ 2,000 QA (총 60K), 고유 이미지 ≥ 1,000/과목 | **타겟 모델 pass@4 난이도 필터 추가**, 2개 독립 분류기 합의 유지 |
| **3단계 (세부분야 커버리지)** | 2026-10-14~16 | **183개 (과목, 세부분야) 쌍별 ≥ 100 고유 이미지 QA** + 균등 샘플링 | `mmmu_train_20261016` 릴리스 확정 (**47,426 QA**, 183/183 분야 대표 확보, 140/183 ≥ 100 달성) |
| **4단계 (STEM/수학 확장)** | 2026-11-15 | 고난도 서술형(open-ended) 수학·CS 추론 데이터 보강 | `mmmu_expansion_mmk12_20261115` 추가 (**9,434 QA**, `FanqingM/MMK12` Apache-2.0, GPU 자동 검수) |

- **최종 통합 학습 풀**: **56,860 QA** (`47,426 + 9,434`), **52,758개 이미지 파일** (`43,335 + 9,423`, ID 중복 0건).

---

## 3. 다중 원천 데이터 라우팅 전략

단일 데이터셋에 의존할 때 발생하는 도메인 편향(예: 수학 80% 편중)을 피하기 위해, **공통 멀티도메인 백본 + 6대 학문 영역별 전용 보강원** 구조를 설계했다.

### 3.1 공통 백본
- **`guanyu615/BMMR`** (8,236건 채택): 교재·시험·퀴즈 기반 300개 UNESCO 분야 멀티모달 문항 및 풀이 포함.
- **`OpenDataArena/MMFineReason-1.8M`** (430건 채택): 전체가 수학 79.4%로 편중되어 있어 통째로 섞지 않고, `source == "BMMR"` 검증 서브셋만 선별 도입.
- **`HuggingFaceM4/FineVision`** (870건 채택): 4개 품질 지표(`relevance`, `visual_dependency`, `image_correspondence`, `formatting`)가 **모두 4점 이상**인 고품질 QA만 채택.
- **`neulab/CulturalGround`** (2,728건 채택): Art, Art Theory, Geography, History, Literature 등 인문·예술 시각 지식 보강.

### 3.2 6대 분야별 전용 보강 데이터
| MMMU 대분류 | 포함 과목 (30개) | 주요 전용 보강 원천 (채택 건수) |
|---|---|---|
| **Health & Medicine** | Basic Medical Science, Clinical Medicine, Diagnostics & Lab, Pharmacy, Public Health | `RadGenome/PMC-VQA` (19,797), `PubMedQA` (6) |
| **Science & Math** | Agriculture, Biology, Chemistry, Geography, Math, Physics | `MMInstruction/ArxivQA` (9,878), `FanqingM/MMK12` (9,434), `Project-AgML/AgroBench` (325), `DiagramQG` (124), `MMSci` (60) |
| **Tech & Engineering** | Architecture & Eng, CS, Electronics, Energy & Power, Materials, Mechanical Eng | `ArxivQA`, `BMMR`, `MMK12`, `StackExchangeOriginalPublishedQA` (22), `CoSyn-400K` (7) |
| **Art & Design** | Art, Art Theory, Design, Music | `MusiXQA` (2,224), `CulturalGround`, `AesMMIT` (257), `FSFR/FakeClue` (146), `Graphic-design-evaluation` (146), `CiQi-VQA` (97), `Q-Instruct-DB` (47), `RuFashion-2M` (32), `ArtVision`/`ArtCoT`/`ArtQuest` |
| **Business** | Accounting, Economics, Finance, Manage, Marketing | `ChartQA` (1,342), `TAT-HQA` (91), `HiTab` (61), `TAT-DQA` (24), `FineVision` |
| **Humanities & Social** | History, Literature, Psychology, Sociology | `CulturalGround`, `PersonaVLM-Dataset` (106), `QuAIL` (96), `QuALITY` (68), `RACE` (65), `VQAonline` (48), `ACLUE` (33), `TFQ-Data-Full` (31) |

- **의도적 배제**: 오염 위험 및 품질 편차가 큰 `VisualWebInstruct`, `LLaVA-CoT`는 배정량을 0으로 고정했다.

---

## 4. 7단계 Fail-Closed 품질 및 오염 차단 게이트

모든 후보 문항은 `training/build_dataset.py` 및 `training/screen_benchmarks.py`에서 아래 7개 게이트를 **전부** 통과해야 채택된다.

### Gate 1. 5중 멀티모달 벤치마크 오염 제거 (Decontamination)
검사 대상은 **`MMMU_dev`, `MMMU_validation`, `MMMU_test`, `MMMU-Pro`** 4개 전체이며, 텍스트와 이미지 양쪽에서 5가지 기준을 모두 통과해야 한다.

| 검사 항목 | 알고리즘 / 구현 | 통과 기준 (차단 임계값) |
|---|---|---|
| **텍스트 완전 일치** | NFKC 정규화 + casefold + 공백 정규화 후 `question + options` 비교 | `exact_match == False` |
| **텍스트 유사 중복** | 5-gram 문자 shingle, 128-permutation MinHash LSH (32 bands × 4 rows) | `minhash_jaccard_max < 0.80` |
| **이미지 내 텍스트(OCR)** | 이미지 OCR 결과의 13-word-gram 집합 매칭 | `ocr_13gram_match == False` |
| **이미지 지각 해시** | 32×32 grayscale 2D-DCT 상위 8×8 기반 64-bit pHash Hamming 거리 | `phash_hamming_min > 6` |
| **이미지 의미 임베딩** | CLIP ViT-B/32 정규화 벡터 코사인 유사도 | `embedding_cosine_max < 0.97` |

### Gate 2. 타겟 모델 Pass@4 난이도 필터링 (Information-Gain Filtering)
- 고정 타겟 모델(`Qwen/Qwen3-VL-4B-Instruct`, rev `ebb281ec...`)에 공식 샘플링 파라미터(`temperature=0.7, top_p=0.8, top_k=20, max_tokens=2048`)로 문항당 **4회 독립 추론**을 수행했다.
- **4/4 정답 문항 전면 제외**: 이미 모델이 확실히 맞히는 쉬운 문항을 걸러내어 학습 컴퓨트를 실제 약점 보강에 집중했다.
- **0/4 ~ 3/4 정답 문항만 채택** (미해결·채점 불명 행은 제외).
  - `20261016` (47,426건): `0/4` 40,049건 (84.4%) · `1/4` 2,439건 (5.1%) · `2/4` 2,586건 (5.5%) · `3/4` 2,352건 (5.0%)
  - `20261115 MMK12` (9,434건): `0/4` 8,557건 (90.7%) · `1/4` 303건 (3.2%) · `2/4` 297건 (3.1%) · `3/4` 277건 (2.9%)
- **오픈형 채점 정밀화 (MMK12)**: 규칙 파서 1차 판정 후 미판정 응답은 `Qwen/Qwen3.8-27B` semantic judge(`confidence >= 0.9`, `quality_issue == False`)로 검수했다.

### Gate 3. 이중 독립 분류기 합의 기반 과목·세부분야 라벨링
- 서로 다른 **2개 이상 독립 분류기**(주축: `Qwen/Qwen3-VL-4B-Instruct` 이미지+텍스트 분류 + `Qwen/Qwen3-1.7B` 텍스트 분류)가 **동일한 MMMU 과목**을 선택하고 **양쪽 모두 confidence ≥ 0.85**일 때만 채택했다.
- 세부 분야(183개 subfield) 라벨 역시 독립 투표 합의분만 부착했다(`20261016` 17,230건 + `MMK12` 4,937건 = 22,167건).

### Gate 4. 시각 의존성 검증 (Visual Dependency)
- 이미지 없이 텍스트만으로 정답을 맞힐 수 있는 문항(text-only solvable QA)을 모델 기반 시각 의존성 검증으로 배제했다.

### Gate 5. 전역 이미지 중복 상한 및 바이트 무결성
- BK-Tree 기반 DCT pHash 근사 중복 클러스터링(Hamming ≤ 6)으로 동일/유사 이미지를 전역 그룹화하고, **동일 지각 이미지 그룹당 최대 2개 QA**로 상한을 제한했다.
- 모든 이미지의 파일 존재, 디코딩 가능 여부, `SHA-256` 해시를 검증했다. (`20261016`: 원본 이미지 42,375장 + 저작권 허용 원문 지문 결정론적 렌더링 960장 = 43,335장; `MMK12`: 원본 이미지 9,423장)

### Gate 6. 문항 형식·정답 정합성 및 제외 레지스트리
- 객관식(2~26개 선택지)의 정답 기호가 실제 선택지 범위와 일치하는지 엄격 검증하고, `true-false` 및 `TAT-DQA/HQA` 타입 참조 정답 형식을 별도 검증했다.
- 정답 충돌이나 선택지 키 모호성이 확인된 문항은 `training/source_qa_exclusions.json`에 등록해 영구 배제했다.

### Gate 7. 라이선스 및 프로비넌스 관리
- 42개 원천 데이터셋(MMK12 포함 43개)의 라이선스(MIT, Apache-2.0, CC-BY-4.0, CC-BY-SA, CC-BY-NC 등)를 개별 추적했다.
- 비상업(NC) 라이선스 원천이 포함되어 있으므로 **내부 연구·과제 평가 전용**으로 제한하고 외부 재배포를 금지한다.

---

## 5. 최종 데이터셋 정량 현황

### 5.1 30개 과목별 분포 (`총 56,860 QA`)

| 과목 (Subject) | `20261016` 기본 | `20261115` MMK12 | 합계 | 과목 (Subject) | `20261016` 기본 | `20261115` MMK12 | 합계 |
|---|---:|---:|---:|---|---:|---:|---:|
| Accounting | 311 | 0 | 311 | Finance | 523 | 3 | 526 |
| Agriculture | 899 | 0 | 899 | Geography | 1,025 | 0 | 1,025 |
| Architecture & Eng. | 340 | 1 | 341 | History | 1,123 | 0 | 1,123 |
| Art | 1,013 | 1 | 1,014 | Literature | 338 | 0 | 338 |
| Art Theory | 413 | 0 | 413 | Manage | 299 | 4 | 303 |
| Basic Medical Science | 2,056 | 0 | 2,056 | Marketing | 101 | 0 | 101 |
| Biology | 1,465 | 0 | 1,465 | Materials | 242 | 0 | 242 |
| Chemistry | 934 | 0 | 934 | Math | 7,541 | 9,178 | 16,719 |
| Clinical Medicine | 2,034 | 0 | 2,034 | Mechanical Eng. | 673 | 0 | 673 |
| Computer Science | 1,581 | 235 | 1,816 | Music | 2,385 | 0 | 2,385 |
| Design | 347 | 0 | 347 | Pharmacy | 387 | 0 | 387 |
| Diagnostics & Lab Med. | 13,684 | 0 | 13,684 | Physics | 5,429 | 2 | 5,431 |
| Economics | 757 | 2 | 759 | Psychology | 342 | 0 | 342 |
| Electronics | 437 | 7 | 444 | Public Health | 349 | 1 | 350 |
| Energy & Power | 308 | 0 | 308 | Sociology | 90 | 0 | 90 |
| | | | | **총계** | **47,426** | **9,434** | **56,860** |

### 5.2 문항 유형 · 이미지 · 내장 풀이(Rationale) 현황
- **문항 유형**: 객관식(MCQ) **41,312건** (72.7%) / 서술·단답형(Open) **15,517건** (27.3%: 기본 6,083 + MMK12 9,434) / 진위형(True-False) **31건** (0.1%)
- **다중 이미지 문항**: 단일 이미지 56,361건, 다중 이미지(2~14장) **499건**
- **주요 이미지 유형**: 의료영상/스캔/병리(18,909+), 그래프·차트(11,073), 기하도형(8,931), 다이어그램(7,202), 악보(2,327), 사진/풍경/현미경/표 등 전 영역 포괄
- **내장 풀이(Rationale) 보유량**: 총 **18,167건** (`ArxivQA` 9,867 · `BMMR` 8,293 · `CoSyn-400K` 7)
  - 길이 분포: 중앙값 **362자**, 90백분위 **966자** (컴팩트하고 밀도 높은 단계별 풀이로, SFT 및 StructReward RL의 핵심 참조 자원)
- **세부 분야 커버리지**: 183/183 전체 세부 분야에 최소 1건 이상 확보, **140/183 분야가 100건 하한 충족** (희소 예술이론·사회학·경영 세부 분야 43개는 균등 가중치 샘플링으로 학습 비중 보정)

---

## 6. 선행 실험 감사에서 발견한 데이터 한계와 후속 SFT + RL 학습 설계 반영 사항

기존 학습 실험 및 실패 분석(`evaluation_runs/20261103_train_distribution_audit`, `docs/research/2026-10-27-existing-train-rationales.md`)에서 확인된 **5가지 핵심 병목**을 정리하고, 이를 후속 학습 파이프라인에서 어떻게 해결할지 명시한다.

1. **단답 강제 시스템 프롬프트의 CoT 파괴 문제**
   - 기존 `train_messages.jsonl`의 99.5%(42,488/42,684)가 `"Give only the final answer. Do not explain."` → `"C"` 단답 타겟으로 포맷되어 있었다.
   - 이를 그대로 SFT하면 모델의 단계별 추론(CoT) 능력이 붕괴해 MMMU-Pro 점수가 오히려 하락(36.7% → 25.8%)했다.
   - **해결**: 내장 풀이 18,167건과 검증된 단계별 풀이(concise structured CoT + 마지막 줄 `Answer: $LETTER` 형식)를 사용해 **추론 허용(중립) 시스템 프롬프트**로 재구성해 SFT한다.
2. **토큰 상한 잘림(Truncation) 및 파서 오추출 문제**
   - 베이스 모델의 공식 CoT 응답 중 **31~35%**가 `max_tokens=2048`에 걸려 `Answer: X`를 출력하지 못하고 끊긴다. 이때 공식 파서가 랜덤 추측하거나 본문 대명사 `"I"`를 선택지 `I`로 오인한다.
   - **해결**: SFT 단계에서 간결하고 완결된 단계별 추론(평균 200~600 토큰)을 학습시키고, RL 단계에서 **형식 보상 + 길이/완결성 게이트 + 일관성 보상(StructReward)**으로 2,048 토큰 내 완결률을 ~99%로 끌어올린다.
3. **4지선다 편중 vs MMMU-Pro 10지선다(Standard10) 및 Vision 보드 격차**
   - 수집된 객관식의 94.4%(38,982/41,312)가 4지선다이며, MMMU-Pro는 **10지선다(`A~J`)** 및 **질문·선택지가 이미지 안에 렌더링된 Vision 트랙**으로 평가된다.
   - **해결**:
     - 서술형(Open) 15,517건(MMK12 + 기본)을 함께 활용해 4지선다 찍기 편향을 억제하고 진짜 추론력을 강화한다.
     - 외부 학습 데이터(leakage-free)의 일부를 **Vision 보드 형태(질문+보기를 이미지로 합성 렌더링)** 및 **확장 선택지(A~J 리매핑/오답 보강)** 형태로 변환해 학습에 포함함으로써 MMMU-Pro Standard10과 Vision 모두에 대한 전이 성능을 극대화한다.
4. **과목 불균형 및 일부 라벨 노이즈**
   - 원천 풀은 `Math`(16,719)와 `Diagnostics`(13,684) 비중이 전체의 53.5%를 차지하며, `Marketing`(101)·`Sociology`(90) 등 희소 과목과의 격차가 크다. 또한 소수 문항에서 분류기 라벨 불일치(예: 천체물리 그래프가 Marketing으로 분류)가 관찰되었다.
   - **해결**: 과목별 균등/완화 가중치 샘플링(temperature-balanced sampling)을 적용하고, SFT/RL 투입 전 소스별 품질·정합성 필터링을 수행한다.
5. **0/4 고난도 편중과 RL Zero-Variance 문제**
   - 전체 풀의 85.5%가 베이스 모델 기준 `0/4` 정답 문항이다. GRPO 계열 RL에서는 그룹 내 모든 롤아웃이 틀리면(all-zero reward) advantage가 0이 되어 학습 신호가 소실된다.
   - **해결**: **2단계(Lightweight Knowledge/CoT SFT → StructReward Gated GRPO RL)** 파이프라인을 채택한다. SFT로 도메인 지식과 간결한 추론 구조를 먼저 주입해 정답 가능 영역을 넓힌 뒤, RL에서는 **혼합 난이도(1/4~3/4 + SFT 후 해결 가능해진 0/4)** 문항에 대해 **StructReward(정답 게이트 + 참조 풀이 단계 정렬 밀집 보상 + 형식/일관성 보상)**를 적용해 안정적으로 추론 정책을 최적화한다.

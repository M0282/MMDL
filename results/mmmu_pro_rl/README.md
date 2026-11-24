# MMMU-Pro Zero-Leakage SFT & StructReward-GRPO 전체 평가 결과 (3,460문항)

모든 실험은 `Qwen/Qwen3-VL-4B-Instruct` (`rev ebb281ec...`, BF16)를 대상으로 **Zero-Leakage 외부 56,860 QA (30과목 · 183세부분야)** 풀에서 수행되었으며, MMMU-Pro 전체 `Standard10` (1,730) + `Vision` (1,730) = **3,460문항**을 SGLang BF16과 공식 채점 로직으로 평가했습니다.

| 모델 / 실험 조건 | `max_tokens` | `pp` | 프롬프트 | Standard10 | Vision | **Overall** | Δ vs 8K / 2K Base | 잘림률 (S / V) |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| Base (`eval_base_2k_pp0`) | 2048 | 0.0 | `official_cot` | 43.99% | 41.68% | 42.84% | — / -1.78%p | 37.3% / 40.2% |
| **Base 2K (`eval_base_2k`)** | 2048 | 1.5 | `official_cot` | 45.43% | 43.82% | **44.62%** | — / 0.00%p | 33.5% / 36.3% |
| Base 2K (`eval_base_2k_budget`) | 2048 | 1.5 | `budget_cot` | 47.86% | 44.34% | 46.10% | — / +1.48%p | 19.5% / 21.9% |
| Stage 1 Off-Policy SFT (`eval_sft_v1_2k`) | 2048 | 1.5 | `official_cot` | 43.29% | 40.69% | 41.99% | — / -2.63%p | 12.4% / 12.7% |
| **Stage 2a RL v1 2K (`eval_rl_v1_2k`)** | 2048 | 1.5 | `official_cot` | **47.34%** | **45.95%** | **46.65%** | — / **+2.03%p** | 23.9% / 26.3% |
| Stage 2b RL v2 2K (`eval_rl_v2_2k`) | 2048 | 1.5 | `official_cot` | 46.36% | 44.91% | 45.63% | — / +1.01%p | 23.6% / 25.4% |
| **Base 8K (`eval_base_8k`)** | 8192 | 1.5 | `official_cot` | 51.27% | 47.46% | **49.37%** | 0.00%p / +4.75%p | 17.6% / 19.1% |
| **★ Stage 2a RL v1 8K (`eval_rl_v1_8k`, pass@1 우승)** | 8192 | 1.5 | `official_cot` | **53.12%** | **49.94%** | **51.53%** | **+2.16%p / +6.91%p** | **8.0% / 8.9%** |
| Stage 3 Hard-STEM RL v4 8K (`eval_rl_v4_8k`) | 8192 | 1.5 | `official_cot` | 51.91% | 48.79% | 50.35% | +0.98%p / +5.73%p | 7.8% / 8.7% |
| **RL v1 8K (`maj@5` TTS)** | 8192 | 1.5 | `official_cot` | 52.89% | 52.37% | **52.63%** | +3.26%p / +8.01%p | 3.5% / 3.5% |

## 핵심 결론
1. **Off-Policy SFT 망각 재현 및 해부**: Stage 1 SFT는 잘림률을 34.9%→12.5%로 줄였으나 Base 완결 문항에서 `-190건` 망각으로 `41.99% (-2.63%p)` 하락.
2. **Base 직결 On-Policy StructReward-GRPO 돌파**: Vision+Merger 100% 동결 + Base 직결 Reverse KL(`β=0.02`) + 30과목 균형 풀 학습으로 2K(`+2.03%p`)와 8K(`+2.16%p vs 8K Base / +6.91%p vs 2K Base`) 모두에서 순수 가중치 성능 향상 달성.
3. **8K 포렌식 분해**: Base가 8K에서도 무한루프로 잘리던 635문항을 `+116건 (+18.3%p)` 회복하면서, Vision 완결 문항 망각은 `-0.64%p`로 사실상 0에 가깝게 억제.

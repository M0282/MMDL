#!/usr/bin/env python
"""
0단계 진단 실험 — transformers 배치 버전

run_stage0_vllm.py와 조건(A/B/C), 프롬프트, 채점(공식 파서 + 수정 파서), 출력 형식이 같다.
다른 점: 추론을 transformers로 하되 여러 문항을 한 번에(batch) 생성해서 속도를 올린다.
  - baseline(run_mmmu_eval.py)과 같은 백엔드라 조건 A를 51.89%와 비교하기 더 쉽다.
  - presence_penalty는 baseline의 LogitsProcessor를 배치용으로 바꾼 것 (EOS 제외 규칙 동일).
  - 배치 샘플링이라 seed가 같아도 batch_size가 다르면 결과가 조금 달라질 수 있음 → batch_size도 meta에 기록.
  - GPU 메모리 부족(OOM)이 나면 그 배치를 반으로 나눠 다시 시도.

사용 예:
  python scripts/run_stage0_hf.py --condition A --seed 42
  python scripts/run_stage0_hf.py --condition B --seed 42 --limit_per_subject 2 --out_dir results/stage0_smoke
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_mmmu_eval import (  # noqa: E402
    DATA_REVISION, MODEL_REVISION, PRESENCE_PENALTY, SAMPLING_RECIPE, SUBJECTS, gather_images,
)
from run_stage0_vllm import CONDITIONS, MIN_PIXELS, build_prompt, grade, load_mmmu_split  # noqa: E402

import torch  # noqa: E402
from transformers import AutoProcessor, LogitsProcessor, Qwen3VLForConditionalGeneration  # noqa: E402


class BatchPresencePenalty(LogitsProcessor):
    """생성된 토큰 중 한 번이라도 나온 토큰의 logit에서 고정값을 뺀다 (문항별로 따로). EOS는 제외."""

    def __init__(self, penalty, prompt_len, eos_token_ids):
        self.penalty = penalty
        self.prompt_len = prompt_len
        if eos_token_ids is None:
            eos_token_ids = []
        elif isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]
        self.eos = torch.tensor(list(eos_token_ids), dtype=torch.long)

    def __call__(self, input_ids, scores):
        gen = input_ids[:, self.prompt_len:]
        if gen.shape[1] == 0:
            return scores
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(1, gen, True)
        if len(self.eos):
            mask[:, self.eos.to(scores.device)] = False
        return scores - self.penalty * mask.to(scores.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=list(CONDITIONS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model_path", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--model_revision", default=MODEL_REVISION)
    ap.add_argument("--data_root", default="MMMU/MMMU")
    ap.add_argument("--data_revision", default=DATA_REVISION)
    ap.add_argument("--out_dir", default="results/stage0")
    ap.add_argument("--split", default="validation", choices=["validation", "test"],
                    help="test는 최종 평가용 (MMMU 정답 공개본, 10,500문항)")
    ap.add_argument("--limit_per_subject", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--tag", default="")
    ap.add_argument("--adapter_path", default="",
                    help="LoRA 어댑터 폴더 (예: adapter_B). 주면 base 모델에 붙여서(merge) 평가")
    args = ap.parse_args()

    cond = CONDITIONS[args.condition]
    name = f"{args.condition}_seed{args.seed}" + (f"_{args.tag}" if args.tag else "")
    os.makedirs(args.out_dir, exist_ok=True)
    jsonl_path = os.path.join(args.out_dir, f"{name}.jsonl")
    meta_path = os.path.join(args.out_dir, f"{name}.meta.json")

    print(f"[조건 {args.condition}] {cond} seed={args.seed} batch={args.batch_size} split={args.split}")

    def load_subject(subj):
        ds = load_mmmu_split(args.data_root, subj, args.data_revision, split=args.split)
        if args.limit_per_subject:
            ds = ds.select(range(min(args.limit_per_subject, len(ds))))
        return [(subj, s) for s in ds]

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, revision=args.model_revision, dtype=torch.bfloat16, device_map="cuda")
    if args.adapter_path:
        # LoRA 어댑터를 붙인 뒤 가중치에 합쳐서(merge) 일반 모델처럼 생성 → 속도 손해 없음
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model = model.merge_and_unload()
        print(f"  [어댑터] {args.adapter_path} 적용 (merge 완료)")
    model.eval()
    processor = AutoProcessor.from_pretrained(
        args.model_path, revision=args.model_revision, min_pixels=MIN_PIXELS, max_pixels=cond["max_pixels"])
    processor.tokenizer.padding_side = "left"
    ip = processor.image_processor
    eff = {k: getattr(ip, k, None) for k in ("min_pixels", "max_pixels", "size", "patch_size")}
    print(f"  [확인] 실제 적용된 이미지 설정: {eff}")
    eos = model.generation_config.eos_token_id

    def prepare(samples, start_idx=0):
        # 프롬프트 텍스트 + 이미지. 이미지 수·크기가 비슷한 것끼리 묶어서 padding 낭비를 줄임
        prepared = []
        for idx, (subj, s) in enumerate(samples, start=start_idx):
            prompt, is_mc, letters, index2ans = build_prompt(s, cond["prompt_mode"])
            images = gather_images(s)
            content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
            text = processor.apply_chat_template([{"role": "user", "content": content}],
                                                 tokenize=False, add_generation_prompt=True)
            size = sum(im.width * im.height for im in images)
            prepared.append(dict(idx=idx, subj=subj, s=s, is_mc=is_mc, letters=letters,
                                 index2ans=index2ans, images=images, text=text, size=size))
        return prepared

    torch.manual_seed(args.seed)
    results = {}
    image_tokens_logged = False

    def run_batch(batch):
        nonlocal image_tokens_logged
        imgs = [im for p in batch for im in p["images"]]
        inputs = processor(text=[p["text"] for p in batch], images=imgs or None,
                           padding=True, return_tensors="pt").to(model.device)
        if not image_tokens_logged and "image_grid_thw" in inputs:
            g = inputs["image_grid_thw"][0].tolist()
            print(f"  [확인] 첫 이미지 grid(t,h,w)={g} → 패치 {g[0] * g[1] * g[2]}개")
            image_tokens_logged = True
        plen = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=cond["max_new_tokens"],
                                 logits_processor=[BatchPresencePenalty(PRESENCE_PENALTY, plen, eos)],
                                 **SAMPLING_RECIPE)
        eos_set = set([eos] if isinstance(eos, int) else (eos or []))
        pad = processor.tokenizer.pad_token_id
        for p, seq in zip(batch, out[:, plen:].tolist()):
            toks = []
            for t in seq:
                if t in eos_set:
                    break
                toks.append(t)
            hit_eos = len(toks) < len(seq) and seq[len(toks)] in eos_set
            toks = [t for t in toks if t != pad]
            text = processor.decode(toks, skip_special_tokens=True).strip()
            results[p["idx"]] = (text, len(toks), not hit_eos and len(seq) >= cond["max_new_tokens"])

    def run_safe(batch):
        try:
            run_batch(batch)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(batch) == 1:
                p = batch[0]
                results[p["idx"]] = ("", 0, False)
                print(f"  [OOM] {p['s']['id']} 1문항도 안 들어감 → 오답 처리")
                return
            h = len(batch) // 2
            print(f"  [OOM] 배치 {len(batch)} → {h}+{len(batch) - h}로 나눠 재시도")
            run_safe(batch[:h])
            run_safe(batch[h:])

    def run_group(prepared, label=""):
        order = sorted(prepared, key=lambda p: (len(p["images"]), p["size"]))
        g0 = time.time()
        for i in range(0, len(order), args.batch_size):
            run_safe(order[i:i + args.batch_size])
            done = min(i + args.batch_size, len(order))
            el = time.time() - g0
            print(f"  {label}[{done}/{len(order)}] 경과 {el / 60:.1f}분, "
                  f"남은 예상 {el / done * (len(order) - done) / 60:.1f}분", flush=True)

    counts = {"n": 0, "correct": 0, "correct_off": 0}

    def write(prepared, f):
        for p in prepared:
            response, n_tok, truncated = results.pop(p["idx"])
            pred, correct, pred_off, correct_off, seg = grade(
                response, cond["prompt_mode"], p["is_mc"], p["letters"], p["index2ans"], p["s"]["answer"])
            counts["n"] += 1
            counts["correct"] += correct
            counts["correct_off"] += correct_off
            f.write(json.dumps({
                "id": p["s"]["id"], "subject": p["subj"], "is_mc": p["is_mc"], "gold": p["s"]["answer"],
                "response": response, "final_segment": seg if seg != response else None,
                "parsed_pred": pred, "correct": correct,
                "parsed_pred_official": pred_off, "correct_official": correct_off,
                "n_gen_tokens": n_tok, "truncated": truncated,
                "finish_reason": "length" if truncated else "stop", "error": None,
            }, ensure_ascii=False) + "\n")
        f.flush()

    t0 = time.time()
    if args.split == "validation":
        # validation: 900문항을 한 번에 정렬·배치 (0단계·SFT 평가와 완전히 같은 방식 → 결과 재현)
        samples = [x for subj in SUBJECTS for x in load_subject(subj)]
        print(f"  {len(samples)}문항 로드")
        prepared = prepare(samples)
        run_group(prepared)
        with open(jsonl_path, "w") as f:
            write(prepared, f)
    else:
        # test(10,500문항): 이미지를 한꺼번에 메모리에 올리지 않도록 과목별로 처리.
        # 과목이 끝날 때마다 jsonl에 저장 → 중간에 끊겨도 다시 실행하면 끝난 과목은 건너뜀
        done_subj = set()
        if os.path.exists(jsonl_path):
            for line in open(jsonl_path):
                r = json.loads(line)
                done_subj.add(r["subject"])
                counts["n"] += 1
                counts["correct"] += r["correct"]
                counts["correct_off"] += r["correct_official"]
            # 마지막 과목은 중간에 끊겼을 수 있으니 그 과목 줄을 지우고 다시 함
            last = None
            lines = open(jsonl_path).read().splitlines()
            if lines:
                last = json.loads(lines[-1])["subject"]
                n_last = sum(1 for l in lines if json.loads(l)["subject"] == last)
                expected = len(load_subject(last))
                if n_last < expected:
                    keep = [l for l in lines if json.loads(l)["subject"] != last]
                    open(jsonl_path, "w").write("\n".join(keep) + ("\n" if keep else ""))
                    done_subj.discard(last)
                    dropped = [json.loads(l) for l in lines if json.loads(l)["subject"] == last]
                    counts["n"] -= len(dropped)
                    counts["correct"] -= sum(r["correct"] for r in dropped)
                    counts["correct_off"] -= sum(r["correct_official"] for r in dropped)
            print(f"  [재시작] 이미 끝난 과목 {len(done_subj)}개 건너뜀")
        idx0 = 0
        with open(jsonl_path, "a") as f:
            for k, subj in enumerate(SUBJECTS):
                if subj in done_subj:
                    continue
                samples = load_subject(subj)
                prepared = prepare(samples, idx0)
                idx0 += len(samples)
                run_group(prepared, label=f"({k + 1}/30 {subj}) ")
                write(prepared, f)
                del prepared, samples
                el = time.time() - t0
                print(f"  == {subj} 끝 · 누적 {counts['n']}문항 · 경과 {el / 60:.1f}분", flush=True)
    elapsed = time.time() - t0
    n_samples, n_correct, n_correct_off = counts["n"], counts["correct"], counts["correct_off"]

    meta = {
        "condition": args.condition, "split": args.split, **cond, "min_pixels": MIN_PIXELS, "seed": args.seed,
        "model_path": args.model_path, "model_revision": args.model_revision,
        "data_revision": args.data_revision, "backend": "transformers-batched",
        "adapter_path": args.adapter_path or None,
        "batch_size": args.batch_size, "effective_image_processor": str(eff),
        "n_samples": n_samples, "n_correct": n_correct,
        "micro_acc": n_correct / n_samples, "micro_acc_official_parser": n_correct_off / n_samples,
        "elapsed_sec": round(elapsed, 1), "gpu": torch.cuda.get_device_name(0),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "torch": torch.__version__, "sampling": {**SAMPLING_RECIPE, "presence_penalty": PRESENCE_PENALTY},
    }
    import transformers
    meta["transformers"] = transformers.__version__
    with open(meta_path, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\n[{name}] 수정 파서 {n_correct / n_samples * 100:.2f}% | 공식 파서 "
          f"{n_correct_off / n_samples * 100:.2f}%  ({elapsed / 60:.1f}분, peak VRAM {meta['peak_vram_gb']}GB)")
    print(f"저장: {jsonl_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""StructReward: Deterministic, auditable gated process + outcome reward for multimodal RL.

Inspired by Li et al. (2026) "StructReward: Efficient Structured Process Rewards for
Self-Correcting Multimodal Reasoning":
    R(y) = G_ans(y) * (1.0 + alpha * R_process(y, ref_steps))
           + R_format(y)
           + R_consistency(y)
           - R_length(y)

Key properties:
- Zero external LLM judge overhead during RL rollouts.
- Final-answer correctness gate G_ans prevents partial-step reward hacking on wrong answers.
- Deterministic step-level alignment (numeric, symbolic, lexical anchors) against verified
  reference reasoning steps.
- Endpoint consistency check penalizes option-letter contradictions ("option D ... Answer: E").
- Works seamlessly for 4-10 option MC (Standard & Vision-board) AND Open-ended STEM.
"""
from __future__ import annotations

import math
import re
from typing import Any

NUM_RE = re.compile(r"(?<![A-Za-z0-9_])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?%?")
WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_\-]{2,}")
STEP_SPLIT_RE = re.compile(
    r"(?m)(?:^|\n)\s*(?:#{1,4}\s+|[-*]\s+)?(?:\*\*)?(?:Step\s*\d+[^\n:*]*:?(?:\*\*)?|(?:\d+)\.\s+(?:\*\*)?)",
    re.I,
)
ANSWER_LINE_RE = re.compile(r"(?im)^\s*(?:\*\*)?Answer(?:\*\*)?\s*:\s*.*$")

STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "are", "was", "were", "have", "has",
    "had", "not", "but", "can", "will", "would", "could", "should", "which", "what", "when",
    "where", "there", "their", "they", "them", "then", "than", "into", "onto", "over", "under",
    "about", "above", "below", "between", "through", "during", "before", "after", "since",
    "because", "while", "therefore", "thus", "hence", "also", "only", "just", "very", "more",
    "most", "some", "each", "both", "such", "does", "did", "done", "being", "been", "here",
    "given", "shown", "image", "figure", "question", "option", "options", "answer", "correct",
    "step", "visual", "observation", "analysis", "reasoning", "verification", "final", "choice",
    "based", "using", "find", "value", "result", "check", "let", "see", "look", "shows",
}


def normalize_number(s: str) -> str:
    s = s.replace(",", "").strip()
    is_pct = s.endswith("%")
    if is_pct:
        s = s[:-1]
    try:
        v = float(s)
        if abs(v) < 1e-12:
            return "0%" if is_pct else "0"
        if abs(v - round(v)) < 1e-6:
            res = str(int(round(v)))
        else:
            res = f"{v:.4g}"
        return res + ("%" if is_pct else "")
    except ValueError:
        return s


def extract_step_anchors(step_text: str) -> tuple[set[str], set[str]]:
    """Extract normalized numerical/symbolic anchors and domain keywords from a step."""
    nums = {normalize_number(m.group(0)) for m in NUM_RE.finditer(step_text)}
    words = {w.lower() for w in WORD_RE.findall(step_text) if w.lower() not in STOPWORDS}
    return nums, words


def split_reasoning_steps(text: str) -> list[str]:
    """Split a response (excluding terminal Answer: line) into structured steps."""
    # Strip terminal Answer: line
    body = ANSWER_LINE_RE.sub("", text).strip()
    if not body:
        return []
    parts = [p.strip() for p in STEP_SPLIT_RE.split(body) if p and len(p.strip()) >= 12]
    if len(parts) >= 2:
        return parts[:8]
    # Fallback to non-empty paragraphs
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if len(p.strip()) >= 12]
    if len(paras) >= 2:
        return paras[:8]
    lines = [l.strip() for l in body.splitlines() if len(l.strip()) >= 12]
    return lines[:8] if lines else ([body] if len(body) >= 12 else [])


def build_reference_steps(rationale: str) -> list[dict[str, list[str]]]:
    """Precompute compact step anchors from a verified reference rationale."""
    steps = split_reasoning_steps(rationale)
    refs = []
    for st in steps:
        nums, words = extract_step_anchors(st)
        if nums or words:
            refs.append({"nums": sorted(nums)[:12], "words": sorted(words)[:16]})
    return refs


def compute_process_alignment(response: str, ref_steps: list[dict[str, list[str]]]) -> float:
    """Deterministic monotonic step-level alignment score in [0.0, 1.0]."""
    if not ref_steps:
        return 0.0
    gen_steps = split_reasoning_steps(response)
    if not gen_steps:
        return 0.0

    gen_anchors = [extract_step_anchors(st) for st in gen_steps]
    # Order-aware soft matching: each ref step matches best gen step at or after previous match
    matched_scores = []
    cursor = 0
    for ref in ref_steps:
        r_nums = set(ref.get("nums") or [])
        r_words = set(ref.get("words") or [])
        best_s = 0.0
        best_idx = cursor
        for j in range(cursor, len(gen_anchors)):
            g_nums, g_words = gen_anchors[j]
            num_sim = (len(r_nums & g_nums) / len(r_nums)) if r_nums else None
            word_sim = (len(r_words & g_words) / max(1, min(len(r_words), 6))) if r_words else None
            if num_sim is not None and word_sim is not None:
                sim = 0.65 * min(1.0, num_sim) + 0.35 * min(1.0, word_sim)
            elif num_sim is not None:
                sim = min(1.0, num_sim)
            elif word_sim is not None:
                sim = min(1.0, word_sim)
            else:
                sim = 0.0
            if sim > best_s:
                best_s = sim
                best_idx = j
        matched_scores.append(best_s)
        if best_s >= 0.35:
            cursor = min(len(gen_anchors) - 1, best_idx)

    # Anti-spam precision damper: penalize if generated response has excessive unrelated tokens
    raw_cov = sum(matched_scores) / len(ref_steps)
    n_words = len(response.split())
    if n_words > 550:
        raw_cov *= max(0.5, 550.0 / n_words)
    return max(0.0, min(1.0, raw_cov))


def parse_strict_mc_answer(response: str, valid_choices: list[str]) -> tuple[str | None, bool]:
    """Strict MC answer parser (NEVER uses random fallback!).
    Returns (parsed_letter_or_None, has_exact_terminal_format).
    """
    if not response or not response.strip():
        return None, False
    valid_set = {c.upper() for c in valid_choices}
    lines = [l.strip() for l in response.strip().splitlines() if l.strip()]
    last_line = lines[-1] if lines else ""
    # Check exact terminal format: Answer: X, **Answer: X**, **Answer:** X, **Answer**: (X), Answer: $X$
    m_term = re.fullmatch(
        r"(?:\*\*)?(?i:Answer)(?:\*\*)?\s*:\s*(?:\*\*\s*)?\$?\(?([A-Za-z])\)?\$?(?:\*\*)?\.?",
        last_line,
    )
    if m_term and m_term.group(1).upper() in valid_set:
        return m_term.group(1).upper(), True

    # Check anywhere in response for explicit Answer: $LETTER (uppercase letter only to avoid matching "a"/"i")
    matches = re.findall(
        r"(?i:(?:final\s+)?answer)(?:\*\*)?\s*(?i:is|:)\s*(?:\*\*\s*)?\$?\(?([A-Z])\)?\$?(?:\*\*)?(?=[.,;:\s]|$)",
        response,
    )
    for cand in reversed(matches):
        if cand in valid_set:
            return cand, False
    return None, False


def _extract_balanced_boxed(text: str) -> list[str]:
    """Extract all \\boxed{...} contents with proper balanced-brace handling."""
    results: list[str] = []
    needle = "\\boxed{"
    start = 0
    while True:
        idx = text.find(needle, start)
        if idx == -1:
            break
        i = idx + len(needle)
        depth = 1
        while i < len(text) and depth > 0:
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            i += 1
        if depth == 0:
            results.append(text[idx + len(needle) : i - 1].strip())
            start = i
        else:
            break
    return results


def _unwrap_latex_command(s: str, cmd: str) -> str:
    """Replace \\cmd{content} -> content with balanced braces."""
    needle = f"\\{cmd}{{"
    while needle in s:
        idx = s.find(needle)
        i = idx + len(needle)
        depth = 1
        while i < len(s) and depth > 0:
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
            i += 1
        if depth != 0:
            break
        inner = s[idx + len(needle) : i - 1]
        s = s[:idx] + inner + s[i:]
    return s


def _strip_math_delimiters(s: str) -> str:
    """Iteratively strip markdown bold, trailing period, $...$, $$...$$, \\(...\\), \\[...\\], \\boxed{...}."""
    s = s.strip()
    for _ in range(6):
        prev = s
        s = s.strip()
        if s.startswith("**") and s.endswith("**") and len(s) >= 4:
            s = s[2:-2].strip()
        if s.endswith(".") and not re.search(r"\d\.$", s):
            s = s[:-1].strip()
        if s.startswith("$$") and s.endswith("$$") and len(s) >= 4:
            s = s[2:-2].strip()
        elif s.startswith("$") and s.endswith("$") and len(s) >= 2:
            s = s[1:-1].strip()
        elif s.startswith("\\(") and s.endswith("\\)") and len(s) >= 4:
            s = s[2:-2].strip()
        elif s.startswith("\\[") and s.endswith("\\]") and len(s) >= 4:
            s = s[2:-2].strip()
        if s.startswith("\\boxed{") and s.endswith("}"):
            boxed = _extract_balanced_boxed(s)
            if len(boxed) == 1 and s == f"\\boxed{{{boxed[0]}}}":
                s = boxed[0].strip()
        if s == prev:
            break
    return s


def normalize_open_ans(s: str) -> str:
    s = _strip_math_delimiters(str(s))
    # Normalize common unicode math symbols
    replacements = [
        ("\u2212", "-"),
        ("\u2013", "-"),
        ("\u2014", "-"),
        ("\u00d7", "\\times"),
        ("\u00f7", "\\div"),
        ("\u2264", "\\le"),
        ("\u2265", "\\ge"),
        ("\u2260", "\\ne"),
        ("\u03c0", "\\pi"),
        ("\u221e", "\\infty"),
        ("\u222a", "\\cup"),
        ("\u2229", "\\cap"),
        ("\\%", "%"),
        ("\\dfrac", "\\frac"),
        ("\\tfrac", "\\frac"),
        ("\\leq", "\\le"),
        ("\\geq", "\\ge"),
        ("\\neq", "\\ne"),
        ("\\cdot", "*"),
        ("\\times", "*"),
        ("^{\\circ}", "°"),
        ("^\\circ", "°"),
        ("\\degree", "°"),
        ("\\left", ""),
        ("\\right", ""),
        ("\\qquad", ""),
        ("\\quad", ""),
        ("\\,", ""),
        ("\\;", ""),
        ("\\:", ""),
        ("\\!", ""),
        ("\\ ", ""),
    ]
    for old, new in replacements:
        s = s.replace(old, new)
    for cmd in ("text", "mathrm", "mathbf", "textbf", "operatorname", "mathit"):
        s = _unwrap_latex_command(s, cmd)
    # Remove any remaining balanced \boxed{...} inside expression
    s = _unwrap_latex_command(s, "boxed")
    s = s.replace(" ", "").lower()
    s = s.replace("+\\infty", "\\infty")
    s = s.rstrip(".")
    return s


def parse_strict_open_answer(response: str) -> tuple[str | None, bool]:
    if not response or not response.strip():
        return None, False
    lines = [l.strip() for l in response.strip().splitlines() if l.strip()]
    last_line = lines[-1] if lines else ""
    m_term = re.match(
        r"^(?:\*\*)?(?i:Answer)(?:\*\*)?\s*:\s*(?:\*\*\s*)?(.+?)(?:\*\*)?$",
        last_line,
    )
    if m_term:
        val = m_term.group(1).strip()
        if val:
            return val, True
    m_any = re.findall(r"(?i:Answer)(?:\*\*)?\s*:\s*(?:\*\*\s*)?(.+)", response)
    if m_any:
        val = m_any[-1].strip().rstrip("*").strip()
        if val:
            return val, False
    m_box = _extract_balanced_boxed(response)
    if m_box:
        return m_box[-1].strip(), False
    return None, False


def _read_brace_group(s: str, pos: int) -> tuple[str, int] | None:
    if pos >= len(s) or s[pos] != "{":
        return None
    i = pos + 1
    depth = 1
    while i < len(s) and depth > 0:
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return s[pos + 1 : i - 1], i


def _eval_simple_math(expr: str) -> tuple[float, bool] | None:
    """Safe, pure-Python numeric evaluator for simple math answers (no eval()).
    Returns (float_value, is_percent) or None if not purely numeric/constant.
    Handles signed numbers, thousands commas, \\frac{a}{b}, a/b, \\sqrt{a}, \\pi, ^, %.
    """
    if not expr:
        return None
    s = expr.strip()
    is_pct = False
    if s.endswith("%"):
        is_pct = True
        s = s[:-1]
    if s.endswith("°"):
        s = s[:-1]
    if not s:
        return None
    # Strip outer parens only if they match each other (NOT intervals like (-2,5))
    if "," in s:
        # Only allow thousands separators like 1,200 or 1,234,567.5
        if re.fullmatch(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", s):
            s = s.replace(",", "")
        else:
            return None

    def _eval_inner(t: str) -> float | None:
        t = t.strip()
        if not t:
            return None
        # Strip matching outer parens
        while t.startswith("(") and t.endswith(")"):
            depth = 0
            balanced = True
            for idx, ch in enumerate(t):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0 and idx < len(t) - 1:
                        balanced = False
                        break
            if balanced and depth == 0:
                t = t[1:-1].strip()
            else:
                break
        if not t:
            return None
        # Leading sign
        if t.startswith("+"):
            return _eval_inner(t[1:])
        if t.startswith("-"):
            # Only treat as unary minus if not split at top-level + or - later
            sub = _eval_inner(t[1:])
            if sub is not None:
                return -sub
        # Direct float
        try:
            v = float(t)
            return v if math.isfinite(v) else None
        except ValueError:
            pass
        if t == "\\pi":
            return math.pi
        # \frac{a}{b}
        if t.startswith("\\frac{"):
            g1 = _read_brace_group(t, len("\\frac"))
            if g1 is not None:
                num_s, nxt = g1
                g2 = _read_brace_group(t, nxt)
                if g2 is not None and g2[1] == len(t):
                    den_s, _ = g2
                    nv, dv = _eval_inner(num_s), _eval_inner(den_s)
                    if nv is not None and dv is not None and abs(dv) > 1e-15:
                        return nv / dv
        # \sqrt{a}
        if t.startswith("\\sqrt{"):
            g1 = _read_brace_group(t, len("\\sqrt"))
            if g1 is not None and g1[1] == len(t):
                v = _eval_inner(g1[0])
                if v is not None and v >= 0:
                    return math.sqrt(v)
        # Top-level binary operators: split outside () and {}
        def split_top(op_chars: str) -> list[tuple[str, str]]:
            parts: list[tuple[str, str]] = []
            depth = 0
            last = 0
            op = "+"
            for i, ch in enumerate(t):
                if ch in "({":
                    depth += 1
                elif ch in ")}":
                    depth -= 1
                elif depth == 0 and ch in op_chars and i > 0:
                    # Skip scientific notation e.g. 1e-5
                    if ch in "+-" and t[i - 1] in "eE" and i >= 2 and (t[i - 2].isdigit() or t[i - 2] == "."):
                        continue
                    parts.append((op, t[last:i]))
                    op = ch
                    last = i + 1
            if parts:
                parts.append((op, t[last:]))
            return parts

        add_parts = split_top("+-")
        if len(add_parts) > 1:
            acc = 0.0
            for op, seg in add_parts:
                v = _eval_inner(seg)
                if v is None:
                    return None
                acc = acc + v if op == "+" else acc - v
            return acc

        mul_parts = split_top("*/")
        if len(mul_parts) > 1:
            acc = 1.0
            for idx, (op, seg) in enumerate(mul_parts):
                v = _eval_inner(seg)
                if v is None:
                    return None
                if idx == 0 or op == "*":
                    acc *= v
                else:
                    if abs(v) <= 1e-15:
                        return None
                    acc /= v
            return acc

        pow_parts = split_top("^")
        if len(pow_parts) == 2:
            base_v = _eval_inner(pow_parts[0][1])
            exp_s = pow_parts[1][1]
            if exp_s.startswith("{") and exp_s.endswith("}"):
                exp_s = exp_s[1:-1]
            exp_v = _eval_inner(exp_s)
            if base_v is not None and exp_v is not None and abs(exp_v) <= 20:
                try:
                    res = base_v ** exp_v
                    return float(res) if isinstance(res, (int, float)) and math.isfinite(res) else None
                except Exception:
                    return None

        # Implicit multiplication: e.g. 2\sqrt{3}, 3\pi, 2\frac{1}{2} (careful: mixed fraction vs product)
        m_imp = re.fullmatch(r"(\d+(?:\.\d+)?)(\\sqrt\{.+\}|\\pi)", t)
        if m_imp:
            c = _eval_inner(m_imp.group(1))
            rhs = _eval_inner(m_imp.group(2))
            if c is not None and rhs is not None:
                return c * rhs
        return None

    val = _eval_inner(s)
    if val is None or not math.isfinite(val):
        return None
    return val, is_pct


def _strip_var_prefix(norm: str) -> str:
    """Strip single-variable assignment prefix like 'x=' when RHS has no '='."""
    m = re.fullmatch(r"[a-z]=(.+)", norm)
    if m and "=" not in m.group(1):
        return m.group(1)
    return norm


def check_open_match(pred: str | None, gold: str) -> bool:
    if not pred:
        return False
    p_norm = normalize_open_ans(pred)
    g_norm = normalize_open_ans(gold)
    if not p_norm or not g_norm:
        return False
    if p_norm == g_norm:
        return True

    # Single-letter MC answer inside open pool (e.g. gold='A', pred='(A)' or 'A')
    p_let = re.fullmatch(r"\(?([a-z])\)?", p_norm)
    g_let = re.fullmatch(r"\(?([a-z])\)?", g_norm)
    if p_let and g_let and p_let.group(1) == g_let.group(1):
        return True

    # Allow optional variable prefix 'x=5' vs '5'
    p_rhs = _strip_var_prefix(p_norm)
    g_rhs = _strip_var_prefix(g_norm)
    if p_rhs == g_rhs and (p_rhs != p_norm or g_rhs != g_norm):
        return True

    # Numeric equivalence via safe math parser
    ep = _eval_simple_math(p_rhs)
    eg = _eval_simple_math(g_rhs)
    if ep is not None and eg is not None:
        fp, ppct = ep
        fg, gpct = eg
        if ppct == gpct:
            return abs(fp - fg) <= 1e-4 * max(1.0, abs(fg))
        # Handle 50% == 0.5 equivalence
        fp_eff = fp / 100.0 if ppct else fp
        fg_eff = fg / 100.0 if gpct else fg
        return abs(fp_eff - fg_eff) <= 1e-4 * max(1.0, abs(fg_eff))
    return False


def check_endpoint_consistency(response: str, parsed_letter: str | None, valid_choices: list[str]) -> float:
    """Check if the final Answer: X matches the letter cited in the concluding reasoning sentence."""
    if not parsed_letter:
        return -0.10
    valid_set = {c.upper() for c in valid_choices}
    body = ANSWER_LINE_RE.sub("", response).strip()
    tail = body[-320:] if len(body) > 320 else body
    # Look for explicit concluding choice mentions in tail (strictly uppercase choice letter!)
    mentions = re.findall(
        r"(?i:option|choice|letter|corresponds\s+to|matches|select)\s+(?:\*\*)?\(?([A-Z])\)?(?:\*\*)?(?=[.,;:\s]|$)",
        tail,
    )
    valid_mentions = [m for m in mentions if m in valid_set]
    if not valid_mentions:
        return 0.0
    if valid_mentions[-1] == parsed_letter:
        return 0.15
    return -0.15


def score_struct_reward(
    response: str,
    gold: str,
    question_type: str,
    valid_choices: list[str] | None = None,
    ref_steps: list[dict[str, list[str]]] | None = None,
    finish_reason: str = "stop",
    completion_tokens: int = 0,
    process_weight: float = 0.50,
    len_ramp_start: int = 1050,
    trunc_pen: float = 0.35,
) -> dict[str, Any]:
    """Compute full StructReward decomposition for one rollout."""
    truncated = finish_reason == "length"

    if question_type == "multiple-choice":
        choices = valid_choices or [chr(65 + i) for i in range(10)]
        parsed, exact_fmt = parse_strict_mc_answer(response, choices)
        g_ans = 1.0 if (parsed is not None and parsed == gold) else 0.0
        r_cons = check_endpoint_consistency(response, parsed, choices)
    else:
        parsed, exact_fmt = parse_strict_open_answer(response)
        g_ans = 1.0 if check_open_match(parsed, gold) else 0.0
        r_cons = 0.10 if (parsed is not None and exact_fmt) else (-0.10 if parsed is None else 0.0)

    # Structured thinking check (encourages multi-step reasoning without needing <think> tags)
    steps = split_reasoning_steps(response)
    has_reasoning = len(steps) >= 2 and len(response.split()) >= 25
    r_format = (0.15 if (exact_fmt and not truncated) else 0.0) + (0.10 if has_reasoning else 0.0)

    # Deterministic step-level process alignment (GATED by g_ans!)
    r_proc_raw = compute_process_alignment(response, ref_steps or [])
    r_proc_gated = g_ans * process_weight * r_proc_raw

    # Concise length shaping: punish truncation & rambling > len_ramp_start tokens
    r_len_pen = 0.0
    if truncated:
        r_len_pen += trunc_pen
    elif completion_tokens > len_ramp_start:
        r_len_pen += min(0.30, 0.30 * (completion_tokens - len_ramp_start) / 550.0)

    total = g_ans + r_proc_gated + r_format + r_cons - r_len_pen
    return {
        "reward": round(float(total), 5),
        "g_ans": g_ans,
        "r_process_raw": round(float(r_proc_raw), 4),
        "r_process_gated": round(float(r_proc_gated), 4),
        "r_format": round(float(r_format), 4),
        "r_consistency": round(float(r_cons), 4),
        "r_len_pen": round(float(r_len_pen), 4),
        "parsed": parsed,
        "exact_format": exact_fmt,
        "truncated": truncated,
        "num_steps": len(steps),
    }

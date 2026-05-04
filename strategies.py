"""
All 8 judge strategies.

S2, S3, and S4 are not used due to compute constraints.
"""

import re
from collections import Counter


def _extract(text: str) -> str | None:
    """Parse 'A', 'B', or 'tie' from judge output. Returns None if unparseable."""
    text = text.strip()
    if text.upper() in ("A", "B"):
        return text.upper()
    if text.lower() in ("tie", "neither", "equal"):
        return "tie"

    m = re.search(
        r"(?:verdict|answer|final answer|winner|better response)\s*[:\-]\s*([\"']?)([AB]|tie|neither|equal)\1",
        text, re.IGNORECASE,
    )
    if m:
        raw = m.group(2).upper()
        return "tie" if raw in ("TIE", "NEITHER", "EQUAL") else raw

    m = re.search(r"response\s+([AB])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Case-sensitive for A/B to avoid matching "a" (article) as verdict "A".
    # "tie"/"neither"/"equal" are still matched case-insensitively via a separate pass.
    tokens = re.findall(r'\b([AB])\b', text) + re.findall(r'\b(tie|neither|equal)\b', text, re.IGNORECASE)
    # Re-sort by position so "last" is still the rightmost token in the text.
    positions = [(text.rfind(t), t) for t in tokens]
    if positions:
        last = max(positions, key=lambda x: x[0])[1].upper()
        return "tie" if last in ("TIE", "NEITHER", "EQUAL") else last

    return None


def _extract_robust(text: str, generate_fn, messages: list[dict]) -> str:
    """Two-pass extraction: direct parse, then one follow-up if needed."""
    verdict = _extract(text)
    if verdict is not None:
        return verdict
    followup = messages + [
        {"role": "assistant", "content": text},
        {"role": "user", "content": 'Please state your final verdict as a single word: "A", "B", or "tie".'},
    ]
    text2, _ = generate_fn(followup, max_new_tokens=10)
    return _extract(text2) or "tie"



def _pairwise_messages(question, resp_a, resp_b, system=None, instruction=None):
    body = (
        f"[Question]\n{question}\n\n"
        f"[Response A]\n{resp_a}\n\n"
        f"[Response B]\n{resp_b}\n\n"
        + (instruction or 'Which response is better? Output ONLY: "A", "B", or "tie"')
    )
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": body})
    return msgs


_SYS_IMPARTIAL = (
    "You are an impartial evaluator. Judge based on accuracy, "
    "helpfulness, and relevance. Do not let position or length influence you."
)
_SYS_COT = (
    "You are an impartial, expert evaluator. "
    "Think step by step before giving your final verdict."
)
_COT_INSTRUCTION = (
    "First explain your reasoning, then give your final verdict as exactly: "
    '"Verdict: A", "Verdict: B", or "Verdict: tie".'
)
_FLIP = {"A": "B", "B": "A", "tie": "tie"}


# ── S0a: Minimal baseline 

def judge_s0a(question, response_a, response_b, generate_fn):
    msgs = [{"role": "user", "content": (
        "Given the question and two responses below, which response is better?\n\n"
        f"[Question]\n{question}\n\n"
        f"[Response A]\n{response_a}\n\n"
        f"[Response B]\n{response_b}\n\n"
        'Output ONLY: "A", "B", or "tie"'
    )}]
    out, tokens = generate_fn(msgs, max_new_tokens=20)
    return _extract_robust(out, generate_fn, msgs), tokens


# ── S0b: Enhanced baseline 

def judge_s0b(question, response_a, response_b, generate_fn):
    msgs = _pairwise_messages(question, response_a, response_b, system=(
        "You are an impartial, expert evaluator. "
        "Judge responses based on accuracy, helpfulness, and relevance to the question. "
        "Do not favor either response based on length, position, or formatting. "
        'If both are equal in quality, say "tie".'
    ))
    out, tokens = generate_fn(msgs, max_new_tokens=20)
    return _extract_robust(out, generate_fn, msgs), tokens


# ── S1: Swap-and-Resolve 
def judge_s1(question, response_a, response_b, generate_fn):
    def _call(ra, rb):
        msgs = _pairwise_messages(question, ra, rb, system=_SYS_IMPARTIAL)
        out, tok = generate_fn(msgs, max_new_tokens=20)
        return _extract_robust(out, generate_fn, msgs), tok

    v1, t1 = _call(response_a, response_b)
    v2_raw, t2 = _call(response_b, response_a)
    v2 = _FLIP.get(v2_raw, "tie")
    return (v1 if v1 == v2 else "tie"), t1 + t2


# ── S2: CoT judge 

def judge_s2(question, response_a, response_b, generate_fn):
    msgs = _pairwise_messages(question, response_a, response_b,
                              system=_SYS_COT, instruction=_COT_INSTRUCTION)
    out, tokens = generate_fn(msgs, max_new_tokens=300)
    return _extract_robust(out, generate_fn, msgs), tokens


# ── S3: Majority vote (k=3, temp=0.7) 

def judge_s3(question, response_a, response_b, generate_fn):
    msgs = _pairwise_messages(question, response_a, response_b, system=_SYS_IMPARTIAL)
    verdicts, total = [], 0
    for _ in range(3):
        out, tok = generate_fn(msgs, max_new_tokens=20, temperature=0.7)
        verdicts.append(_extract_robust(out, generate_fn, msgs))
        total += tok
    counts = Counter(verdicts)
    majority = counts.most_common(1)[0][0]
    return ("tie" if counts.most_common(1)[0][1] == 1 else majority), total


# ── S4: Swap + CoT 

def judge_s4(question, response_a, response_b, generate_fn):
    def _cot_call(ra, rb):
        msgs = _pairwise_messages(question, ra, rb, system=_SYS_COT, instruction=_COT_INSTRUCTION)
        out, tok = generate_fn(msgs, max_new_tokens=300)
        return _extract_robust(out, generate_fn, msgs), tok

    v1, t1 = _cot_call(response_a, response_b)
    v2_raw, t2 = _cot_call(response_b, response_a)
    v2 = _FLIP.get(v2_raw, "tie")
    return (v1 if v1 == v2 else "tie"), t1 + t2


# ── S5: Rubric decomposition 

_RUBRIC_CRITERIA = [
    ("accuracy and factuality",
     "Focus ONLY on which response is more accurate and factually correct."),
    ("helpfulness and completeness",
     "Focus ONLY on which response is more helpful and complete in addressing the question."),
    ("clarity and organization",
     "Focus ONLY on which response is clearer, better organized, and easier to understand."),
]


def judge_s5(question, response_a, response_b, generate_fn):
    verdicts, total = [], 0
    for crit_name, crit_instr in _RUBRIC_CRITERIA:
        msgs = _pairwise_messages(
            question, response_a, response_b,
            system=f"You are an impartial evaluator assessing {crit_name}. {crit_instr} Do not consider other factors.",
            instruction=f'Judging only {crit_name}: which response is better? Output ONLY: "A", "B", or "tie"',
        )
        out, tok = generate_fn(msgs, max_new_tokens=20)
        verdicts.append(_extract_robust(out, generate_fn, msgs))
        total += tok
    counts = Counter(verdicts)
    majority = counts.most_common(1)[0][0]
    return ("tie" if counts.most_common(1)[0][1] == 1 else majority), total


# ── S6: Pointwise scoring 

_SCORE_SYSTEM = (
    "You are an expert evaluator. Rate the quality of the following response "
    "to the given question on a scale of 1 to 10:\n"
    " 1–2  Very poor\n 3–4  Poor\n 5–6  Adequate\n 7–8  Good\n 9–10 Excellent\n"
    "Output ONLY a single integer between 1 and 10. No explanation."
)


def _extract_score(text: str) -> int | None:
    text = text.strip()
    m = re.search(r"(?:score|rating|verdict|answer)\s*[:\-]?\s*(\d+)", text, re.IGNORECASE)
    if m and 1 <= int(m.group(1)) <= 10:
        return int(m.group(1))
    m = re.search(r"(\d+)\s*/\s*10", text)
    if m and 1 <= int(m.group(1)) <= 10:
        return int(m.group(1))
    for n in re.findall(r"\b(\d+)\b", text):
        if 1 <= int(n) <= 10:
            return int(n)
    return None


def _score_one(question, response, generate_fn):
    msgs = [
        {"role": "system", "content": _SCORE_SYSTEM},
        {"role": "user", "content": f"[Question]\n{question}\n\n[Response]\n{response}"},
    ]
    out, tok = generate_fn(msgs, max_new_tokens=10)
    score = _extract_score(out)
    if score is None:
        followup = msgs + [
            {"role": "assistant", "content": out},
            {"role": "user", "content": "Please output only the integer score (1–10)."},
        ]
        out2, tok2 = generate_fn(followup, max_new_tokens=5)
        score = _extract_score(out2)
        tok += tok2
    return score, tok


def judge_s6(question, response_a, response_b, generate_fn):
    sa, ta = _score_one(question, response_a, generate_fn)
    sb, tb = _score_one(question, response_b, generate_fn)
    if sa is None or sb is None:
        return "tie", ta + tb
    if sa > sb:
        return "A", ta + tb
    if sb > sa:
        return "B", ta + tb
    return "tie", ta + tb


# ── S7: Forced-Choice Swap 

def judge_s7(question, response_a, response_b, generate_fn):
    """
    Forced-Choice Swap — 2 calls, no tie option per call.

    Each call removes "tie" from the instruction. If the model abstains,
    a follow-up forces a binary choice. Ties only emerge when the two
    forced choices disagree after position-flipping — a genuine position-bias
    signal, not over-abstention.

    Key difference from S1: S1 allows "tie" per call, letting the judge
    over-abstain on close pairs. S7 forces a direction on every call, so
    the final tie rate reflects real disagreement rather than evasion.
    """
    def _forced_call(ra, rb):
        msgs = _pairwise_messages(question, ra, rb, system=_SYS_IMPARTIAL,
                                  instruction='Which response is better? Output ONLY: "A" or "B"')
        out, tok = generate_fn(msgs, max_new_tokens=20)
        v = _extract(out)
        if v == "tie":
            v = None  # reject tie — prompt removed it as an option
        if v is None:
            followup = msgs + [
                {"role": "assistant", "content": out},
                {"role": "user", "content": 'You must choose one. Output ONLY: "A" or "B"'},
            ]
            out2, tok2 = generate_fn(followup, max_new_tokens=10)
            v = _extract(out2)
            if v == "tie":
                v = None
            tok += tok2
        return (v if v in ("A", "B") else "A"), tok  # last-resort default

    v1, t1 = _forced_call(response_a, response_b)
    v2_raw, t2 = _forced_call(response_b, response_a)
    v2 = _FLIP.get(v2_raw, v2_raw)
    return (v1 if v1 == v2 else "tie"), t1 + t2



STRATEGIES = [
    ("S0a", "S0a: Minimal",          judge_s0a, 1),
    ("S0b", "S0b: Enhanced",         judge_s0b, 1),
    ("S1",  "S1: Swap",              judge_s1,  2),
    ("S2",  "S2: CoT",               judge_s2,  1),
    ("S3",  "S3: Vote(k=3)",         judge_s3,  3),
    ("S4",  "S4: Swap+CoT",          judge_s4,  2),
    ("S5",  "S5: Rubric",            judge_s5,  3),
    ("S6",  "S6: Pointwise",         judge_s6,  2),
    ("S7",  "S7: Forced-Choice Swap",  judge_s7, 2),
]
# key → (display_name, judge_fn, calls_per_judgment)
STRATEGY_MAP = {key: (name, fn, calls) for key, name, fn, calls in STRATEGIES}

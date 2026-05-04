"""
Load pairwise preference datasets.

All loaders return a list of dicts:
  question_id, question, response_a, response_b,
  human_label ("A"|"B"), model_a, model_b, category, turn
"""

import random
from collections import defaultdict
from datasets import load_dataset

DATASETS = ("mt_bench", "chatbot_arena")

_MTBENCH_CATEGORIES = {
    range(81,  91): "writing",
    range(91,  101): "roleplay",
    range(101, 111): "reasoning",
    range(111, 121): "math",
    range(121, 131): "coding",
    range(131, 141): "extraction",
    range(141, 151): "stem",
    range(151, 161): "humanities",
}


def _mt_bench_category(qid: int) -> str:
    for r, cat in _MTBENCH_CATEGORIES.items():
        if qid in r:
            return cat
    return "unknown"


def _conversation_text(conv, idx):
    if idx >= len(conv):
        return ""
    return conv[idx].get("content", "").strip()


def _build_mt_bench_question(row):
    conv_a = row["conversation_a"]
    turn = row["turn"]

    question_turn1 = _conversation_text(conv_a, 0)
    if turn == 1:
        return question_turn1

    response_a_turn1 = _conversation_text(conv_a, 1)
    response_b_turn1 = _conversation_text(row["conversation_b"], 1)
    question_turn2 = _conversation_text(conv_a, 2)
    return (
        "A user had the following conversation with two AI assistants.\n\n"
        "[Round 1]\n"
        f"User: {question_turn1}\n"
        f"Assistant A: {response_a_turn1}\n"
        f"Assistant B: {response_b_turn1}\n\n"
        "[Round 2]\n"
        f"User: {question_turn2}\n\n"
        "Using the full conversation context above, which assistant gave a better response "
        "to the Round 2 user question?"
    )


def _build_mt_bench_pair(row, winner):
    turn = row["turn"]
    response_idx = 2 * turn - 1
    response_a = _conversation_text(row["conversation_a"], response_idx)
    response_b = _conversation_text(row["conversation_b"], response_idx)

    if not response_a or not response_b:
        return None

    return {
        "question_id": row["question_id"],
        "question": _build_mt_bench_question(row),
        "response_a": response_a,
        "response_b": response_b,
        "human_label": "A" if winner == "model_a" else "B",
        "model_a": row["model_a"],
        "model_b": row["model_b"],
        "category": _mt_bench_category(row["question_id"]),
        "turn": turn,
    }


def _load_mt_bench(n, seed, mt_bench_turns):
    include_turns = {1} if mt_bench_turns == "first" else {1, 2}
    ds = load_dataset("lmsys/mt_bench_human_judgments", split="human")

    groups = defaultdict(list)
    for row in ds:
        if row["turn"] in include_turns:
            groups[(row["question_id"], row["turn"], row["model_a"], row["model_b"])].append(row)

    pairs = []
    for rows in groups.values():
        counts = defaultdict(int)
        for r in rows:
            counts[r["winner"]] += 1
        winner = max(counts, key=counts.get)
        if winner == "tie" or counts[winner] <= len(rows) / 2:
            continue
        row = rows[0]
        pair = _build_mt_bench_pair(row, winner)
        if pair is not None:
            pairs.append(pair)

    random.Random(seed).shuffle(pairs)
    return pairs[:n] if n is not None else pairs


def _moderation_flagged(field) -> bool:
    if not field:
        return False
    entries = field if isinstance(field, list) else [field]
    return any(isinstance(e, dict) and e.get("flagged", False) for e in entries)


def _load_chatbot_arena(n, seed):
    ds = load_dataset("lmsys/chatbot_arena_conversations", split="train")

    counts = defaultdict(int)
    pairs = []
    for row in ds:
        counts["total"] += 1

        if row.get("language", "").lower() != "english":
            counts["non_english"] += 1; continue
        if row.get("winner", "") not in ("model_a", "model_b"):
            counts["ties"] += 1; continue
        if row.get("model_a", "") == row.get("model_b", ""):
            counts["same_model"] += 1; continue
        if row.get("toxic", False) or _moderation_flagged(row.get("openai_moderation")):
            counts["moderation"] += 1; continue

        conv_a, conv_b = row.get("conversation_a", []), row.get("conversation_b", [])
        if len(conv_a) < 2 or len(conv_b) < 2:
            counts["short_conv"] += 1; continue

        q, ra, rb = conv_a[0]["content"], conv_a[1]["content"], conv_b[1]["content"]
        if not q.strip() or not ra.strip() or not rb.strip():
            counts["empty_text"] += 1; continue

        counts["kept"] += 1
        pairs.append({
            "question_id": row.get("question_id", row.get("conversation_id", "")),
            "question":    q,
            "response_a":  ra,
            "response_b":  rb,
            "human_label": "A" if row["winner"] == "model_a" else "B",
            "model_a":     row.get("model_a", ""),
            "model_b":     row.get("model_b", ""),
            "category":    "unknown",
        })

    labels = ["non_english", "ties", "same_model", "moderation", "short_conv", "empty_text", "kept"]
    print(f"  ChatBot Arena filter ({counts['total']} rows): " +
          ", ".join(f"{k}={counts[k]}" for k in labels))

    random.Random(seed).shuffle(pairs)
    return pairs[:n] if n is not None else pairs


def load_data(dataset="mt_bench", n=None, seed=42, mt_bench_turns="both"):
    """
    Load pairs from the specified dataset.

    Args:
        dataset : "mt_bench" or "chatbot_arena"
        n       : cap on pairs returned (None = all).
                  For chatbot_arena, passing n=1000 is recommended.
        seed    : shuffle seed for reproducibility
        mt_bench_turns : "first" or "both"
    """
    if dataset == "mt_bench":
        return _load_mt_bench(n, seed, mt_bench_turns=mt_bench_turns)
    if dataset == "chatbot_arena":
        return _load_chatbot_arena(n, seed)
    raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {DATASETS}")

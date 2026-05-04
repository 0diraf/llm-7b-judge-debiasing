"""
All evaluation metrics:
  strict_agreement    — 3-class exact match
  lenient_agreement   — 2-class (ties excluded from both sides)
  cohens_kappa        — chance-corrected agreement - not used
  position_consistency — fraction of pairs with order-invariant verdict
  verbosity_bias_index — VBI = judge_longer_rate - human_longer_rate
  compute_all_metrics — convenience wrapper returning a single dict
"""

from sklearn.metrics import cohen_kappa_score


def strict_agreement(judge: list[str], human: list[str]) -> float:
    return sum(j == h for j, h in zip(judge, human)) / len(judge)


def lenient_agreement(judge: list[str], human: list[str]) -> float | None:
    pairs = [(j, h) for j, h in zip(judge, human) if j != "tie" and h != "tie"]
    if not pairs:
        return None
    return sum(j == h for j, h in pairs) / len(pairs)


def cohens_kappa(judge: list[str], human: list[str]) -> float:
    try:
        return cohen_kappa_score(human, judge)
    except Exception:
        return float("nan")


def position_consistency(verdicts_ab: list[str], verdicts_ba: list[str]) -> float:
    consistent = sum(
        (ab == "tie" and ba == "tie") or (ab == "A" and ba == "B") or (ab == "B" and ba == "A")
        for ab, ba in zip(verdicts_ab, verdicts_ba)
    )
    return consistent / len(verdicts_ab)


def verbosity_bias_index(judge: list[str], human: list[str], len_a: list[int], len_b: list[int]) -> float:
    j_longer, h_longer = [], []
    for j, h, la, lb in zip(judge, human, len_a, len_b):
        if la == lb:
            continue
        longer = "A" if la > lb else "B"
        j_longer.append(1 if j == longer else 0)
        h_longer.append(1 if h == longer else 0)
    if not j_longer:
        return float("nan")
    return sum(j_longer) / len(j_longer) - sum(h_longer) / len(h_longer)


def compute_all_metrics(
    strategy_name: str,
    judge_verdicts: list[str],
    human_labels: list[str],
    len_a: list[int],
    len_b: list[int],
    total_calls: int,
    total_tokens: int,
    verdicts_swapped: list[str] | None = None,
) -> dict:
    n = len(judge_verdicts)
    return {
        "strategy":                strategy_name,
        "n":                       n,
        "strict_agree":            strict_agreement(judge_verdicts, human_labels),
        "lenient_agree":           lenient_agreement(judge_verdicts, human_labels),
        "kappa":                   cohens_kappa(judge_verdicts, human_labels),
        "position_consistency":    position_consistency(judge_verdicts, verdicts_swapped)
                                   if verdicts_swapped is not None else None,
        "vbi":                     verbosity_bias_index(judge_verdicts, human_labels, len_a, len_b),
        "tie_rate":                sum(v == "tie" for v in judge_verdicts) / n,
        "total_calls":             total_calls,
        "avg_tokens_per_judgment": total_tokens / n if n > 0 else 0,
    }

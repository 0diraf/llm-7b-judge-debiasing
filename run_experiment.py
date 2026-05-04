"""
Full-scale runner: all models and all strategies on the complete dataset.

"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data       import load_data, DATASETS
from generate   import load_model, unload_model, generate, count_tokens, MODELS
from metrics    import compute_all_metrics, strict_agreement
from strategies import STRATEGIES, STRATEGY_MAP

COMPACT_BIAS_STRATEGIES = ["S0a", "S0b", "S1", "S5", "S6", "S7"]
DEFAULT_MODEL_KEYS = ["qwen", "mistral", "olmo"]



def _ckpt_path(out_dir, model_key, strategy_key):
    return os.path.join(out_dir, "checkpoints", model_key, f"{strategy_key}.json")


def _ckpt_exists(out_dir, model_key, strategy_key):
    return os.path.exists(_ckpt_path(out_dir, model_key, strategy_key))


def _save_ckpt(out_dir, model_key, strategy_key, data):
    path = _ckpt_path(out_dir, model_key, strategy_key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)  


def _load_ckpt(out_dir, model_key, strategy_key):
    with open(_ckpt_path(out_dir, model_key, strategy_key)) as f:
        return json.load(f)


def run_strategy(strategy_key, pairs):
    """Run one strategy over all pairs using the currently loaded model."""
    _, fn, _ = STRATEGY_MAP[strategy_key]
    verdicts, total_tokens, errors = [], 0, 0
    row_traces = []

    for i, pair in enumerate(pairs):
        call_trace = []

        def traced_generate(messages, max_new_tokens=50, temperature=None):
            output_text, output_tokens = generate(
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            call_trace.append({
                "call_index": len(call_trace) + 1,
                "message_count": len(messages),
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "output_text": output_text,
                "output_tokens": output_tokens,
            })
            return output_text, output_tokens

        try:
            verdict, tokens = fn(
                pair["question"],
                pair["response_a"],
                pair["response_b"],
                traced_generate,
            )
            error = None
        except Exception as e:
            print(f"  [!] Pair {i} error: {e}")
            verdict, tokens = "tie", 0
            errors += 1
            error = str(e)
        verdicts.append(verdict)
        total_tokens += tokens
        row_traces.append({
            "pair_index": i,
            "question_id": pair.get("question_id"),
            "question_preview": pair.get("question", "")[:200],
            "human_label": pair.get("human_label"),
            "model_a": pair.get("model_a"),
            "model_b": pair.get("model_b"),
            "category": pair.get("category"),
            "turn": pair.get("turn"),
            "len_a": pair.get("len_a"),
            "len_b": pair.get("len_b"),
            "verdict": verdict,
            "tokens_used": tokens,
            "n_generate_calls": len(call_trace),
            "calls": call_trace,
            "error": error,
        })
        if (i + 1) % 25 == 0:
            print(f"  {strategy_key}: {i+1}/{len(pairs)} ({(i+1)/len(pairs)*100:.0f}%)")

    if errors:
        print(f"  [!] {errors}/{len(pairs)} errors")
    return verdicts, total_tokens, row_traces



def build_model_results(model_key, strategy_keys, pairs, out_dir):
    human_labels = [p["human_label"] for p in pairs]
    len_a = [p["len_a"] for p in pairs]
    len_b = [p["len_b"] for p in pairs]

    all_verdicts, results = {}, []
    for skey in strategy_keys:
        if not _ckpt_exists(out_dir, model_key, skey):
            print(f"  [skip] no checkpoint for {model_key}/{skey}")
            continue
        ckpt = _load_ckpt(out_dir, model_key, skey)
        sname, _, calls = STRATEGY_MAP[skey]
        all_verdicts[sname] = ckpt["verdicts"]

        metrics = compute_all_metrics(
            strategy_name=sname,
            judge_verdicts=ckpt["verdicts"],
            human_labels=human_labels,
            len_a=len_a, len_b=len_b,
            total_calls=calls * len(pairs),
            total_tokens=ckpt["total_tokens"],
        )
        if skey in ("S1", "S4"):
            metrics["position_consistency"] = 1.0
        results.append(metrics)

    # Per-category breakdown
    cat_breakdown = {}
    for cat in sorted(set(p["category"] for p in pairs)):
        idx = [i for i, p in enumerate(pairs) if p["category"] == cat]
        if len(idx) < 3:
            continue
        cat_human = [human_labels[i] for i in idx]
        cat_breakdown[cat] = {
            sname: strict_agreement([verdicts[i] for i in idx], cat_human)
            for sname, verdicts in all_verdicts.items()
        }

    turn_breakdown = {}
    turns = sorted(set(p.get("turn") for p in pairs if p.get("turn") is not None))
    for turn in turns:
        idx = [i for i, p in enumerate(pairs) if p.get("turn") == turn]
        if not idx:
            continue
        turn_human = [human_labels[i] for i in idx]
        turn_breakdown[str(turn)] = {
            sname: strict_agreement([verdicts[i] for i in idx], turn_human)
            for sname, verdicts in all_verdicts.items()
        }

    return {
        "model_key":       model_key,
        "model_display":   MODELS[model_key]["display"],
        "n_pairs":         len(pairs),
        "results":         results,
        "verdicts":        all_verdicts,
        "human_labels":    human_labels,
        "category_breakdown": cat_breakdown,
        "turn_breakdown":  turn_breakdown,
    }


def merge_all(model_keys, strategy_keys, pairs, out_dir, dataset, mt_bench_turns=None):
    model_results = {}
    for mkey in model_keys:
        print(f"\nBuilding metrics for {mkey}...")
        model_results[mkey] = build_model_results(mkey, strategy_keys, pairs, out_dir)

    full = {"dataset": dataset, "n_pairs": len(pairs),
            "models": model_keys, "strategies": strategy_keys,
            "mt_bench_turns": mt_bench_turns if dataset == "mt_bench" else None,
            "model_results": model_results}
    out_path = os.path.join(out_dir, "full_results.json")
    with open(out_path, "w") as f:
        json.dump(full, f, indent=2)
    print(f"\nFull results → {out_path}")
    return full



def print_table(model_display, results):
    print(f"\n{'═'*70}\n  {model_display}\n{'═'*70}")
    hdr = f"{'Strategy':<18} {'Strict':>7} {'Lenient':>8} {'Kappa':>6} {'PC':>6} {'VBI':>7} {'Ties':>6} {'Tok/J':>7}"
    print(hdr + "\n" + "-" * len(hdr))
    for r in results:
        la  = f"{r['lenient_agree']*100:>7.1f}%" if r['lenient_agree'] is not None else f"{'N/A':>8}"
        kap = f"{r['kappa']:>6.3f}"              if r['kappa'] == r['kappa']          else f"{'NaN':>6}"
        pc  = f"{r['position_consistency']:>6.3f}" if r['position_consistency'] is not None else f"{'N/A':>6}"
        vbi = f"{r['vbi']:>+7.3f}"              if r['vbi'] == r['vbi']               else f"{'NaN':>7}"
        print(f"{r['strategy']:<18} {r['strict_agree']*100:>6.1f}% {la} {kap} {pc} {vbi}"
              f" {r['tie_rate']*100:>5.1f}% {r['avg_tokens_per_judgment']:>7.0f}")
    print("-" * len(hdr))



def main():
    all_keys = [k for k, *_ in STRATEGIES]
    default_out_dir = str(Path(__file__).parent / "results")

    parser = argparse.ArgumentParser()
    parser.add_argument("--models",     nargs="+", default=DEFAULT_MODEL_KEYS, choices=list(MODELS))
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=COMPACT_BIAS_STRATEGIES,
        choices=all_keys,
        help="Compact default: S0a S0b S1 S5 S6 S7. Pass explicit keys to override.",
    )
    parser.add_argument("--dataset",    default="mt_bench", choices=list(DATASETS))
    parser.add_argument("--n_pairs",    type=int,  default=None,
                        help="Pair cap. For chatbot_arena, recommend --n_pairs 1000")
    parser.add_argument(
        "--mt_bench_turns",
        default="both",
        choices=["first", "both"],
        help="For MT-Bench, evaluate only turn 1 or both turns (default: both).",
    )
    parser.add_argument("--out_dir",    default=default_out_dir)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--save_traces", dest="save_traces", action="store_true")
    parser.add_argument("--no_save_traces", dest="save_traces", action="store_false")
    parser.add_argument("--merge_only",   action="store_true",
                        help="Skip inference; just merge checkpoints → full_results.json")
    parser.set_defaults(save_traces=True)
    args = parser.parse_args()

    out_dir = os.path.join(args.out_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    turn_label = f", turns={args.mt_bench_turns}" if args.dataset == "mt_bench" else ""
    print(f"Dataset: {args.dataset}{turn_label}" + (f" (n={args.n_pairs})" if args.n_pairs else " (all)"))
    pairs = load_data(
        dataset=args.dataset,
        n=args.n_pairs,
        seed=args.seed,
        mt_bench_turns=args.mt_bench_turns,
    )
    print(f"Loaded {len(pairs)} pairs")

    if not args.merge_only:
        # Pre-compute token lengths once, using the first model's tokenizer
        print(f"\nPre-computing token lengths with {MODELS[args.models[0]]['display']}...")
        load_model(args.models[0], load_in_4bit=args.load_in_4bit)
        for p in pairs:
            p["len_a"] = count_tokens(p["response_a"])
            p["len_b"] = count_tokens(p["response_b"])

        first = True
        for model_key in args.models:
            if not first:
                load_model(model_key, load_in_4bit=args.load_in_4bit)
            first = False

            print(f"\n{'━'*60}")
            print(f"  Judge: {MODELS[model_key]['display']}  |  {args.dataset}  ({len(pairs)} pairs)")
            print(f"{'━'*60}")

            for skey in args.strategies:
                sname = STRATEGY_MAP[skey][0]
                if _ckpt_exists(out_dir, model_key, skey):
                    print(f"  [skip] {sname} — checkpoint exists")
                    continue

                print(f"\n── {sname} ──")
                t0 = time.time()
                verdicts, total_tokens, row_traces = run_strategy(skey, pairs)
                elapsed = time.time() - t0
                print(f"  Done in {elapsed/60:.1f} min ({len(pairs)/elapsed*60:.0f} pairs/min)")

                ckpt_data = {
                    "dataset": args.dataset, "model_key": model_key,
                    "strategy_key": skey, "n_pairs": len(pairs),
                    "verdicts": verdicts, "total_tokens": total_tokens,
                    "elapsed_sec": elapsed,
                    "trace_schema": "compact_v1",
                    "mt_bench_turns": args.mt_bench_turns if args.dataset == "mt_bench" else None,
                }
                if args.save_traces:
                    ckpt_data["row_traces"] = row_traces
                _save_ckpt(out_dir, model_key, skey, ckpt_data)
                print(f"  Checkpoint saved.")

            unload_model()
    else:
        for p in pairs:
            p.setdefault("len_a", 0)
            p.setdefault("len_b", 0)

    full = merge_all(
        args.models,
        args.strategies,
        pairs,
        out_dir,
        dataset=args.dataset,
        mt_bench_turns=args.mt_bench_turns,
    )
    for mkey in args.models:
        if mkey in full["model_results"]:
            print_table(MODELS[mkey]["display"], full["model_results"][mkey]["results"])

    print("\nDone.")


if __name__ == "__main__":
    main()

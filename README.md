# LLM Judge Bias Experiments

This repo contains a pipeline for evaluating LLM-as-a-judge strategies on pairwise preference datasets. It compares several prompting and scoring strategies against human labels, with a focus on agreement, tie behavior, position sensitivity, verbosity bias, and token cost.


- `data.py` loads MT-Bench human judgments and Chatbot Arena conversations from Hugging Face.
- `strategies.py` defines judge strategies, including baseline prompting, swap-based debiasing, rubric decomposition, pointwise scoring, and forced-choice swap diagnostics.
- `generate.py` loads local Hugging Face causal language models and runs generation.
- `metrics.py` computes agreement, tie rate, verbosity bias, and token-cost metrics.
- `run_experiment.py` runs models and strategies, saves checkpoints, and merges results
- `llm-debiasing.ipynb` contains the results of running the experiments, as well as EDA on the outputs.

The scripts use Hugging Face models and datasets, so the first run may download model weights and dataset files. A CUDA-capable GPU is recommended. Use `--load_in_4bit` for lower-memory runs.

## Example Runs

Run the default strategy set (`S0a S0b S1 S5 S6 S7`) on MT-Bench:

```bash
python run_experiment.py --dataset mt_bench --mt_bench_turns both --models qwen mistral olmo
```

Run on Chatbot Arena with a smaller sample:

```bash
python run_experiment.py --dataset chatbot_arena --n_pairs 1000 --models qwen mistral olmo
```

Run Qwen 14B in 4-bit mode:

```bash
python run_experiment.py --dataset mt_bench --models qwen14b --strategies S0a S5 S7 --load_in_4bit
```

## Outputs

Each run writes checkpoint files under `results/<dataset>/checkpoints/<model>/`. The final merged result file is saved as:

```text
results/<dataset>/full_results.json
```

Checkpoints include verdicts, token counts, compact row-level traces, pair metadata, and any per-row errors so interrupted runs can be resumed without repeating completed model-strategy combinations.

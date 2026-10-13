# Context-Aware Weighted Composite (CAWC): RAG Hallucination Detection

This workspace contains an unsupervised framework for detecting hallucinated tokens in Retrieval-Augmented Generation responses using context-conditioned and context-free language-model distributions. Model caches are excluded from this repository; generate them using the reproduction steps below.


## Primary results: Qwen2.5-7B, seed 42

Seed **42** is the primary run for metric comparisons, cross-domain evaluation, temporal analysis, type and generator breakdowns, failure cases, and ablations. Its saved results are in [main results/seed_42(primary)/](<main results/seed_42(primary)>).

The following values come from that run's `experiment_1_2_table.csv`. Channel names follow the paper (Section 3); the code identifier of each channel is given in brackets.

| Metric or composite | RAGTruth AUROC |
| --- | ---: |
| Token entropy $H$ | 0.6072 |
| Token improbability $U = 1 - P_c(y_t)$ [`selfcheck_score`] | 0.6098 |
| Entropy reduction $\Delta H$ [`information_gain`] | 0.7180 |
| KL divergence $K$ | 0.6719 |
| Confidence drop $D$ | 0.7268 |
| Semantic entropy $E$ | 0.5308 |
| **CAWC** | **0.7519** |

The primary composite has a reported bootstrap 95% interval of **[0.7326, 0.7677]**, F1 **0.1781**, and HaluEval transfer AUROC **0.6523**. The equal-weight composite in `experiment_weight_ablation.csv` has AUROC **0.7315**.

## Five-seed composite experiment

The additional seeds are used only for the five-seed composite experiment. They are not used to replace or average the primary run's other experiment results.

| Seed | Composite RAGTruth AUROC |
| --- | ---: |
| **42 (primary)** | **0.7519** |
| 43 | 0.7470 |
| 100 | 0.7512 |
| 105 | 0.7513 |
| 137 | 0.7532 |
| **Mean ± sample standard deviation** | **0.7509 ± 0.0023** |

These statistics are calculated from the five saved `experiment_1_2_table.csv` files. The standard deviation describes variation across these runs; it is not a confidence interval. Recreate the composite-only aggregate without loading models or caches:

```bash
python run_multiseed_pipeline.py --aggregate-only
```

## Other model runs

| Scoring model | RAGTruth composite AUROC | Saved result |
| --- | ---: | --- |
| Qwen2.5-1.5B | 0.7416 | `Qwen2.5_1.5B/results/seed_100/experiment_1_2_table.csv` |
| Mistral-7B-v0.1 | 0.7324 | `Mistral7B/results/experiment_1_2_table.csv` |
| Llama-2-7B base | 0.7384 | `results_llama2_7B_base/results/experiment_1_2_table.csv` |

The paper reports **0.7412** for the 1.5B detector, the value in that run's `experiment_4_table.csv`; its metric-comparison table lists 0.7416. These models use their own train statistics and implementations, so they are separate model evaluations.

## Method and limitations

The root evaluator normalizes raw metric values against the training population and derives base weights from training-signal standard deviations, with a minimum-weight floor. It combines these metrics with smoothing, interactions, gradients, and pooling. It does not fit a supervised classifier for the CAWC composite. The Mistral evaluator uses a separate implementation with fixed weights.

Temporal results are retrospective. In the primary run, `experiment_9_analysis.txt` reports the composite's maximum AUROC at **t+2**, after onset. The composite also uses centered windows and forward pooling, which read future token scores. These results do not establish causal prediction before hallucination.

The ablation outputs evaluate distinct weighting variants. For example, the leave-one-out table uses a consensus baseline of **0.7551**; removing semantic entropy leaves its rounded AUROC unchanged. These variant results should not be conflated with the primary composite score of **0.7519**.

## Workspace layout

- Root Python files: Qwen2.5-7B metric extraction, cached evaluation, demos, and ablation tools.
- `main results/seed_42(primary)/`: primary experiment artifacts and caches.
- `main results/seed_43/`, `seed_100/`, `seed_105/`, and `seed_137/`: additional runs used for the composite five-seed experiment; other saved outputs are archived.
- `Qwen2.5_1.5B/` and `Mistral7B/`: model-specific code and results.
- `Redeep implementation/`: ReDeEP comparison scripts and remote-run instructions. The external `ReDEeP-ICLR` source tree required by `run_all.sh` is not included.
- `results_llama2_7B_base/`: saved Llama-2 base-model results.

## Reproducing the results

1. **Compute the paired predictions.**
   This runs the detector with and without the retrieved context and writes the per-token caches. It requires a GPU and overwrites existing caches.
   ```bash
   python run_pipeline_v2.py --base-dir .
   ```

2. **Score and evaluate.**
   This computes the CAWC score from the cached predictions and writes the summary and per-experiment `.csv` and `.txt` files to `results/`.
   ```bash
   python run_final_unsupervised.py
   ```
   The `.pkl` caches must be generated locally using step 1.
3. **Score a new example.**
   Scores every token of a response to a new query and context.
   ```bash
   python demo_pipeline.py --query "Tell me about the Eiffel Tower" \
     --context "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It was constructed from 1887 to 1889." \
     --response "The Eiffel Tower is located in Paris, France. It was built in 1995 by architect Frank Lloyd Wright."
   ```

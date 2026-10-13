#!/usr/bin/env python3
"""
Demo Pipeline -- CAWC
Accepts an unseen (query, context, response) triplet and outputs a
token-level hallucination heatmap using the dual forward-pass approach.

Usage (Windows PowerShell -- everything on ONE line):
  python demo_pipeline.py --query "What caused the 2008 financial crisis?" --context "The 2008 crisis was caused by subprime mortgage lending..." --response "The financial crisis was triggered by..."

Evaluator Q&A references:
  IG(t)  = H_0(t) - H_c(t)          ... positive = context helped = faithful
  KL(t)  = Sum_v P_c * log(P_c/P_0) ... distributional shift from context
  CD(t)  = P_0(y_t) - P_c(y_t)      ... negative = context boosted confidence = faithful
"""
import argparse
import numpy as np
import torch

from model_utils import ModelConfig, load_generator_model, dual_forward_pass
from core_metrics import MetricEvaluator


def run_demo(query: str, context: str, response: str):
    """Score a single (query, context, response) triplet."""

    # ── Step 1: Load Mistral-7B (4-bit quantized) ──────────────────────
    print("\n[1/3] Loading Mistral-7B-v0.1 (4-bit)...")
    cfg = ModelConfig(model_name="mistralai/Mistral-7B-v0.1",
                      max_gpu_memory_gib=6)
    model, tokenizer = load_generator_model(cfg)
    ev = MetricEvaluator(tokenizer=tokenizer, generator_model=model)

    # ── Step 2: Dual forward pass ──────────────────────────────────────
    print("[2/3] Running dual forward pass (with-context vs without-context)...")
    P_c, P_0, token_ids = dual_forward_pass(
        model, tokenizer, query, context, response, cfg
    )

    n = len(P_c)
    if n == 0:
        print("ERROR: No tokens scored. Check input length.")
        return

    # ── Step 3: Compute token-level metrics ────────────────────────────
    print(f"[3/3] Computing metrics for {n} tokens...")

    entropy = np.zeros(n)
    ig      = np.zeros(n)
    kl      = np.zeros(n)
    cd      = np.zeros(n)

    for t in range(n):
        m = ev.token_metrics(P_c[t], P_0[t], int(token_ids[t]))
        entropy[t] = m["entropy"]
        ig[t]      = m["ig"]
        kl[t]      = m["kl"]
        cd[t]      = m["cd"]

    # ── Compute CAWC composite (same as run_final_unsupervised.py) ──────
    # Z-normalize each metric using unsupervised train stats, then combine
    CAWC_WEIGHTS = {"cd": 0.38, "ig": 0.35, "kl": 0.15, "entropy": 0.02}
    train_stats = {
        "entropy": (1.0038, 1.0472),
        "ig":      (0.8666, 1.0633),
        "kl":      (0.9200, 1.3809),
        "cd":      (-0.1974, 0.2997),
    }

    # Direction: negate IG and CD so that HIGH = hallucination for all
    directed = {"entropy": entropy, "ig": -ig, "kl": kl, "cd": -cd}

    z_normed = {}
    for mk, arr in directed.items():
        mu, std = train_stats[mk]
        if mk in ["ig", "cd"]:
            mu = -mu
        z = (arr - mu) / std
        z_normed[mk] = 1.0 / (1.0 + np.exp(-1.5 * z))  # sigmoid to [0,1]

    wsum = sum(CAWC_WEIGHTS.values())
    cawc = sum(CAWC_WEIGHTS[k] / wsum * z_normed[k] for k in CAWC_WEIGHTS)

    # ── Output: Token-level metric table ───────────────────────────────
    tokens = tokenizer.convert_ids_to_tokens(token_ids.tolist())

    print(f"\n{'=' * 80}")
    print(f"{'Token':<18} {'H(t)':>8} {'IG':>8} {'KL':>8} {'CD':>8} {'CAWC':>7}")
    print(f"{'':18} {'(nats)':>8} {'(nats)':>8} {'(nats)':>8} {'(prob)':>8}")
    print(f"{'=' * 80}")

    for t in range(n):
        tok = str(tokens[t])[:16] if t < len(tokens) else "?"
        tok = tok.encode("ascii", errors="replace").decode("ascii")

        print(f"{tok:<18} "
              f"{entropy[t]:>8.3f} "
              f"{ig[t]:>8.3f} "
              f"{kl[t]:>8.3f} "
              f"{cd[t]:>8.3f} "
              f"{cawc[t]:>7.3f}")

    # ── Summary statistics ─────────────────────────────────────────────
    suspicious = int(np.sum(ig < -0.1))
    print(f"\n{'=' * 80}")
    print(f"Summary: {n} tokens scored")
    print(f"  Mean IG:   {ig.mean():+.3f} nats  (positive = context helped)")
    print(f"  Mean CD:   {cd.mean():+.4f}       (negative = context boosted confidence)")
    print(f"  Mean CAWC: {cawc.mean():.3f}       (>0.5 = above-average suspicion)")
    print(f"  Tokens with negative IG: {suspicious} / {n}")

    # ── Metric explanations ────────────────────────────────────────────
    print(f"\n{'-' * 72}")
    print("HOW TO READ THIS TABLE:\n")
    print("  H(t)  = Token Entropy = -Sum P_c[t,v] * log P_c[t,v]")
    print("          High H = model is uncertain about this token.\n")
    print("  IG    = Information Gain = H_0(t) - H_c(t)")
    print("          H_0 = entropy WITHOUT context, H_c = entropy WITH context.")
    print("          Positive IG --> context REDUCED uncertainty --> FAITHFUL.")
    print("          Negative IG --> context INCREASED uncertainty --> HALLUCINATION.\n")
    print("  KL    = KL Divergence = Sum P_c[t,v] * log(P_c[t,v] / P_0[t,v])")
    print("          Measures how much the context shifted the probability distribution.")
    print("          High KL = large distributional shift from adding context.\n")
    print("  CD    = Confidence Drop = P_0(y_t) - P_c(y_t)")
    print("          P_0 = probability of the actual token WITHOUT context.")
    print("          P_c = probability of the actual token WITH context.")
    print("          Negative CD --> context BOOSTED confidence --> FAITHFUL.")
    print("          Positive CD --> context HURT confidence --> HALLUCINATION.")
    print(f"{'-' * 72}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CAWC -- Live Hallucination Detection Demo"
    )
    parser.add_argument("--query", required=True,
                        help="The question/query")
    parser.add_argument("--context", required=True,
                        help="The retrieved context passage")
    parser.add_argument("--response", required=True,
                        help="The model's generated response to analyze")
    args = parser.parse_args()

    run_demo(args.query, args.context, args.response)

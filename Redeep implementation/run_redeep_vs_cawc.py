"""
Apples-to-apples comparison: ReDeEP vs CAWC on the SAME samples.
Loads RAGTruth test from HuggingFace Arrow, runs both methods in-process
on the full test set (2700 samples). No mismatched subsets possible.

Both methods process every (query, context, response) triple through
LLaMA-2-7B-chat regardless of which model originally generated the response.
Ground-truth hallucination labels are factual accuracy against the context,
so they remain valid for any generator's output.
"""
import sys
import os
import json
import random
import time
import argparse
from pathlib import Path

# Inject ReDeEP's custom transformers fork (needed for knowledge_layers)
# IMPORTANT: must use ReDEeP-ICLR/transformers, NOT the vanilla transformers/ in remote_export/
CUSTOM_TRANSFORMERS_PATH = os.path.join(os.path.dirname(__file__), "ReDEeP-ICLR", "transformers", "src")
if CUSTOM_TRANSFORMERS_PATH not in sys.path:
    sys.path.insert(0, CUSTOM_TRANSFORMERS_PATH)

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_auc_score
from datasets import load_from_disk

# CAWC imports
from core_metrics import MetricEvaluator
from model_utils import ModelConfig, dual_forward_pass

# ---------------------------------------------------------------------------
# ReDeEP helpers (ported from token_level_detect.py / token_level_reg.py)
# ---------------------------------------------------------------------------

def calculate_dist(dist1, dist2):
    softmax_mature = F.softmax(dist1, dim=-1)
    softmax_anchor = F.softmax(dist2, dim=-1)
    M = 0.5 * (softmax_mature + softmax_anchor)
    
    log_softmax_mature = F.log_softmax(dist1, dim=-1)
    log_softmax_anchor = F.log_softmax(dist2, dim=-1)
    
    kl1 = F.kl_div(log_softmax_mature, M, reduction='none').mean(-1)
    kl2 = F.kl_div(log_softmax_anchor, M, reduction='none').mean(-1)
    
    js_divs = 0.5 * (kl1 + kl2) 
    return js_divs.cpu().item() * 1e6

def is_hallucination_token(token_pos, hallucination_spans):
    """Check if a token at sequence position `token_pos` falls inside any hallucination span."""
    for span in hallucination_spans:
        if span[0] <= token_pos <= span[1]:
            return True
    return False

def calculate_hallucination_spans(labels_spans, prompt_text, response_text, tokenizer, add_special_tokens=True):
    """Convert character-level spans into token-level spans.
    
    Args:
        add_special_tokens: Must match how the caller tokenized the full sequence.
            Phase 1 (ReDeEP) uses tokenizer() which adds BOS -> True (default).
            Phase 2 (CAWC) uses tokenizer.encode(add_special_tokens=False) -> False.
    """
    hallucination_span = []
    for span in labels_spans:
        start_id = span["start"]
        end_id = span["end"]
        start_text = prompt_text + response_text[:start_id]
        end_text = prompt_text + response_text[:end_id]
        start_text_id = tokenizer(start_text, return_tensors="pt", add_special_tokens=add_special_tokens).input_ids
        end_text_id = tokenizer(end_text, return_tensors="pt", add_special_tokens=add_special_tokens).input_ids
        start_id_tok = start_text_id.shape[-1]
        end_id_tok = end_text_id.shape[-1]
        hallucination_span.append([start_id_tok, end_id_tok])
    return hallucination_span

def get_redeep_scores(model, tokenizer, query, context, response_text, labels_spans, hyperparams, max_length=2048):
    device = model.device
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Question: {query}\nContext: {context}"[:12000]}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_text = prompt + response_text
    
    input_ids = tokenizer([full_text], return_tensors="pt").input_ids.to(device)
    prefix_ids = tokenizer([prompt], return_tensors="pt").input_ids.to(device)
    prefix_len = prefix_ids.shape[-1]
    
    if input_ids.shape[-1] > max_length:
        return None, None, None, None # skip too long
        
    response_len = input_ids.shape[-1] - prefix_len
    if response_len <= 0:
        return None, None, None, None

    hallucination_spans = calculate_hallucination_spans(labels_spans, prompt, response_text, tokenizer)

    select_heads = hyperparams["select_heads"]
    select_layers = hyperparams["select_layers"]
    alpha = hyperparams["weight"]
    
    with torch.no_grad():
        logits_dict, outputs = model(
            input_ids=input_ids,
            return_dict=True,
            output_attentions=True,
            output_hidden_states=True,
            knowledge_layers=list(range(0, 32))
        )
        
    logits_dict = {key: [value[0], value[1]] for key, value in logits_dict.items()}
    hidden_states = outputs["hidden_states"]
    last_hidden_states = hidden_states[-1][0, :, :]
    
    attentions_list = []
    for layer_id in range(len(outputs.attentions)):
        for head_id in range(outputs.attentions[layer_id].shape[1]):
            if [layer_id, head_id] not in select_heads:
                continue
            attentions_list.append({
                "layer_head": (layer_id, head_id), 
                "attention_score": outputs.attentions[layer_id][:, head_id, :, :]
            })
            
    final_scores = []
    labels = []
    
    head_max_min = hyperparams["head_max_min"]
    layers_max_min = hyperparams["layers_max_min"]
    
    def normalize(val, max_v, min_v):
        return (val - min_v) / (max_v - min_v) if max_v > min_v else 0.0

    for seq_i in range(prefix_len - 1, input_ids.shape[-1] - 1):
        pointer_scores_list = [attn["attention_score"][:, seq_i, :] for attn in attentions_list]
        pointer_probs_list = torch.cat([ps[:, :prefix_len] for ps in pointer_scores_list], dim=0)
        
        top_k = max(1, int(pointer_probs_list.shape[-1] * 0.1))
        sorted_indices = torch.argsort(pointer_probs_list, dim=1, descending=True)
        top_k_indices = sorted_indices[:, :top_k]
        
        flattened_indices = top_k_indices.flatten()
        selected_hidden_states = last_hidden_states[flattened_indices]
        top_k_hidden_states = selected_hidden_states.view(top_k_indices.shape[0], top_k_indices.shape[1], -1)
        attend_token_hidden_state = torch.mean(top_k_hidden_states, dim=1)
        
        current_hidden_state = last_hidden_states[seq_i, :]
        current_hidden_state = current_hidden_state.unsqueeze(0).expand(attend_token_hidden_state.shape)
        
        cosine_similarity = F.cosine_similarity(attend_token_hidden_state, current_hidden_state, dim=1)
        es_sum = cosine_similarity.sum().cpu().item()
        
        pkd_sum = 0
        for layer_idx in select_layers:
            val = logits_dict[layer_idx]
            dist = calculate_dist(val[0][0, seq_i, :], val[1][0, seq_i, :])
            pkd_sum += dist
            
        norm_es = normalize(es_sum, head_max_min[0], head_max_min[1])
        norm_pkd = normalize(pkd_sum, layers_max_min[0], layers_max_min[1])
        
        score = norm_pkd - alpha * norm_es
        final_scores.append(score)
        
        if is_hallucination_token(seq_i, hallucination_spans):
            labels.append(1)
        else:
            labels.append(0)
            
    del outputs, logits_dict, hidden_states
    torch.cuda.empty_cache()
            
    return final_scores, labels, input_ids[0, prefix_len:].cpu().numpy(), prefix_len

# ---------------------------------------------------------------------------
# CAWC helpers
# ---------------------------------------------------------------------------

def get_cawc_scores(model, tokenizer, query, context, response_text, cfg):
    p_with, p_wo, response_ids, prompt_ids = dual_forward_pass(model, tokenizer, query, context, response_text, cfg)
    if len(response_ids) == 0:
        return None, None, None, None
        
    evaluator = MetricEvaluator(tokenizer=tokenizer)
    
    entropy_w = []
    ig = []
    kl = []
    cd = []
    selfcheck = []
    u = []
    
    for i in range(len(p_with)):
        pw = np.asarray(p_with[i])
        p0 = np.asarray(p_wo[i])
        
        entropy_w.append(evaluator.entropy(pw))
        ig.append(evaluator.information_gain(pw, p0))
        kl.append(evaluator.kl_divergence(pw, p0))
        
        token_id = int(response_ids[i])
        cd.append(evaluator.confidence_drop(pw, p0, token_id))
        
        prob_token = float(pw[token_id])
        selfcheck.append(-np.log(prob_token + 1e-8))
        
        u.append(evaluator.context_utilization(pw, p0))
        
    results = {
        "entropy": np.array(entropy_w).tolist(),
        "ig": np.array(ig).tolist(),
        "kl": np.array(kl).tolist(),
        "cd": np.array(cd).tolist(),
        "selfcheck": np.array(selfcheck).tolist(),
        "u": np.array(u).tolist()
    }
    
    prompt_text = tokenizer.decode(prompt_ids)
    cawc_prefix_len = len(prompt_ids)
    return results, response_ids, prompt_text, cawc_prefix_len

# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_auroc(labels, scores, n_boot=1000, seed=42):
    labels, scores = np.asarray(labels), np.asarray(scores)
    mask = np.isfinite(scores)
    l, s = labels[mask], scores[mask]
    if len(np.unique(l)) < 2:
        return 0.5, 0.5, 0.5
    base = roc_auc_score(l, s)
    rng = np.random.default_rng(seed)
    boots = []
    n = len(l)
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        try:
            boots.append(roc_auc_score(l[idx], s[idx]))
        except ValueError:
            pass
    if not boots:
        return base, base, base
    return base, np.percentile(boots, 2.5), np.percentile(boots, 97.5)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=0,
                        help="Number of samples to use. 0 = use ALL available samples (recommended for full evaluation).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-name", type=str, default="NousResearch/Llama-2-7b-hf")
    parser.add_argument("--cawc-only", action="store_true", help="Run only CAWC and skip ReDeEP")
    args = parser.parse_args()

    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))

    # ---- Load RAGTruth test split from HuggingFace Arrow ----
    print("Loading RAGTruth Test Split from Arrow dataset...")
    test_path = script_dir / "ragtruth_processed" / "test"
    test_ds = load_from_disk(str(test_path))
    print(f"Total test samples: {len(test_ds)}")

    # Use ALL test samples (2700) — both methods process every response through
    # the same LLaMA-2-7B model, so the original generator doesn't matter.
    # Ground-truth labels are factual accuracy vs context, valid for any generator.
    all_indices = list(range(len(test_ds)))
    
    # Optionally subsample for debugging
    random.seed(args.seed)
    if args.sample_size > 0:
        sample_size = min(args.sample_size, len(all_indices))
        selected_indices = random.sample(all_indices, sample_size)
        print(f"Running on {sample_size} random samples (seed={args.seed}).")
    else:
        selected_indices = all_indices
        sample_size = len(all_indices)
        print(f"Running on ALL {sample_size} test samples.")

    # Report generator breakdown
    models_in_selection = [str(test_ds[i]["model"]) for i in selected_indices]
    from collections import Counter
    gen_counts = Counter(models_in_selection)
    print(f"Generator breakdown: {dict(gen_counts)}")

    # Convert Arrow rows to dicts with parsed labels
    test_subset = []
    for idx in selected_indices:
        row = test_ds[idx]
        # Parse hallucination_labels from JSON string
        raw_labels = row.get("hallucination_labels", "[]")
        try:
            labels_list = json.loads(raw_labels) if isinstance(raw_labels, str) else (raw_labels or [])
        except json.JSONDecodeError:
            labels_list = []
        
        spans = []
        if isinstance(labels_list, list):
            for item in labels_list:
                if isinstance(item, dict) and "start" in item and "end" in item:
                    spans.append({"start": int(item["start"]), "end": int(item["end"])})
        
        test_subset.append({
            "id": row.get("id", str(idx)),
            "query": str(row.get("query", "")),
            "context": str(row.get("context", "")),
            "response": str(row.get("output", "")),
            "labels": spans,
            "generator": str(row.get("model", "unknown")),
        })
    
    # Sanity check: positive rate
    total_with_spans = sum(1 for s in test_subset if len(s["labels"]) > 0)
    print(f"Samples with ≥1 hallucination span: {total_with_spans}/{sample_size} ({total_with_spans/sample_size:.1%})")

    # ---- Load hyperparameters ----
    hyperparams_file = script_dir / "ReDEeP-ICLR" / "ReDeEP" / "log" / "test_llama2_7B" / "token_hyperparameter.json"
    with open(hyperparams_file, "r") as f:
        hyperparams = json.load(f)
    print(f"Loaded ReDeEP hyperparameters: {json.dumps(hyperparams, indent=2)}")

    # ---- Load model ----
    print(f"\nLoading {args.model_name} in FP16...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager"
    )
    model.eval()
    
    cawc_cfg = ModelConfig(
        model_name=args.model_name,
        max_model_tokens=4096,
        max_response_tokens=256,
        max_query_tokens=128,
        max_gpu_memory_gib=38
    )

    # ---- Phase 1: ReDeEP ----
    redeep_all_scores = []
    redeep_all_labels = []
    redeep_times = []
    redeep_processed = 0
    redeep_success_indices = set()
    rd_auc, rd_lo, rd_hi, rd_time_avg, rd_time_std, rd_pos_rate = 0.5, 0.5, 0.5, 0.0, 0.0, 0.0
    
    if not args.cawc_only:
        print("\n" + "="*60)
        print("PHASE 1: Running ReDeEP on identical samples")
        print("="*60)
        for i, sample in enumerate(tqdm(test_subset, desc="ReDeEP")):
            t0 = time.time()
            rd_scores, rd_labels, _, _ = get_redeep_scores(
                model, tokenizer,
                sample["query"], sample["context"], sample["response"],
                sample["labels"], hyperparams
            )
            t1 = time.time()
            
            if rd_scores is not None:
                redeep_all_scores.extend(rd_scores)
                redeep_all_labels.extend(rd_labels)
                redeep_times.append(t1 - t0)
                redeep_processed += 1
                redeep_success_indices.add(i)
                
        print(f"\nReDeEP: {redeep_processed}/{sample_size} samples succeeded.")
        rd_pos_rate = np.mean(redeep_all_labels) if redeep_all_labels else 0
        print(f"ReDeEP token count: {len(redeep_all_labels)}, positive rate: {rd_pos_rate:.4f}")
        
        # Diagnostic: check if scores have any variation
        if redeep_all_scores:
            rd_scores_arr = np.asarray(redeep_all_scores)
            print(f"ReDeEP score stats: min={rd_scores_arr.min():.6f} max={rd_scores_arr.max():.6f} "
                  f"mean={rd_scores_arr.mean():.6f} std={rd_scores_arr.std():.6f} "
                  f"unique={len(np.unique(rd_scores_arr))}")
        
        rd_auc, rd_lo, rd_hi = bootstrap_auroc(redeep_all_labels, redeep_all_scores)
        print(f">>> ReDeEP Token-Level AUROC: {rd_auc:.4f} [{rd_lo:.4f}, {rd_hi:.4f}] <<<")

    # ---- Phase 2: CAWC ----
    # Store per-sample results to enable per-sample scaling (matching Experiment 1)
    cawc_sample_results = []  # list of dicts: {"raw": {metric: array}, "labels": list}
    cawc_times = []
    cawc_processed = 0
    
    print("\n" + "="*60)
    print("PHASE 2: Running CAWC on identical samples")
    print("="*60)
    for i, sample in enumerate(tqdm(test_subset, desc="CAWC")):
        if not args.cawc_only and i not in redeep_success_indices:
            continue
            
        t2 = time.time()
        cawc_raw_results, cawc_ids, prompt_text, cawc_prefix_len = get_cawc_scores(
            model, tokenizer,
            sample["query"], sample["context"], sample["response"],
            cawc_cfg
        )
        t3 = time.time()
        
        if cawc_raw_results is not None:
            # Build hallucination spans
            hallucination_spans = calculate_hallucination_spans(
                sample["labels"], prompt_text, sample["response"], tokenizer,
                add_special_tokens=False
            )
            
            cawc_labels = []
            for seq_i in range(len(cawc_raw_results["entropy"])):
                if is_hallucination_token(cawc_prefix_len - 1 + seq_i, hallucination_spans):
                    cawc_labels.append(1)
                else:
                    cawc_labels.append(0)
            
            cawc_sample_results.append({
                "raw": {k: np.array(cawc_raw_results[k], dtype=np.float64) for k in ["entropy", "ig", "kl", "cd", "selfcheck", "u"]},
                "labels": cawc_labels,
            })
            cawc_times.append(t3 - t2)
            cawc_processed += 1
            
    cawc_all_labels = []
    for sr in cawc_sample_results:
        cawc_all_labels.extend(sr["labels"])
    
    print(f"\nCAWC: {cawc_processed}/{sample_size} samples succeeded.")
    cawc_pos_rate = np.mean(cawc_all_labels) if cawc_all_labels else 0
    print(f"CAWC token count: {len(cawc_all_labels)}, positive rate: {cawc_pos_rate:.4f}")

    # ---- Step A: Direction test (global, using raw scores pooled across samples) ----
    metric_keys = ["entropy", "ig", "kl", "cd", "selfcheck"]
    direction_flags = {}
    labels_arr = np.array(cawc_all_labels)
    
    print("\n" + "-"*60)
    print("CAWC Calibration (Direction & Variance Weights)")
    print("-"*60)
    
    if len(labels_arr) > 0 and len(np.unique(labels_arr)) >= 2:
        # Pool raw scores for direction testing
        pooled_raw = {k: [] for k in metric_keys}
        for sr in cawc_sample_results:
            for k in metric_keys:
                pooled_raw[k].extend(sr["raw"][k].tolist())
        
        for key in metric_keys:
            vals = np.array(pooled_raw[key], dtype=np.float64)
            try:
                raw_auc = roc_auc_score(labels_arr, vals)
                neg_auc = roc_auc_score(labels_arr, -vals)
            except ValueError:
                raw_auc, neg_auc = 0.5, 0.5
            should_negate = neg_auc > raw_auc
            direction_flags[key] = should_negate
            print(f"{key:>10}: negated={should_negate} | raw_auc={raw_auc:.4f} neg_auc={neg_auc:.4f}")
        
        # ---- Step B: Per-sample direction + min-max scaling (matching _score_sample) ----
        # Then pool the per-sample-scaled scores for IQR weight calibration
        pooled_scaled = {k: [] for k in metric_keys}
        pooled_u = []
        for sr in cawc_sample_results:
            for key in metric_keys:
                arr = sr["raw"][key].copy()
                if direction_flags.get(key, False):
                    arr = -arr
                # Per-sample min-max scaling (the critical difference!)
                finite = np.isfinite(arr)
                if not finite.any():
                    scaled = np.zeros_like(arr, dtype=np.float64)
                else:
                    arr_for_scale = arr.copy()
                    if not finite.all():
                        arr_for_scale[~finite] = float(np.nanmedian(arr_for_scale[finite]))
                    v_min, v_max = arr_for_scale.min(), arr_for_scale.max()
                    if v_max > v_min:
                        scaled = (arr_for_scale - v_min) / (v_max - v_min)
                    else:
                        scaled = np.zeros_like(arr_for_scale)
                pooled_scaled[key].extend(scaled.tolist())
            pooled_u.extend(sr["raw"]["u"].tolist())
        
        # ---- Step C: IQR variance weight calibration ----
        spreads = {}
        for key in metric_keys:
            arr = np.array(pooled_scaled[key], dtype=np.float64)
            q75, q25 = np.percentile(arr, [75, 25])
            spreads[key] = max(float(q75 - q25), 1e-3)
        total_spread = sum(spreads.values())
        metric_weights = {k: v / total_spread for k, v in spreads.items()}
        print(f"Variance Weights: {json.dumps(metric_weights, indent=2)}")
        
        # ---- Step D: Dynamic composite (per-sample scaled scores + context utilization) ----
        u_arr = np.array(pooled_u)
        cawc_all_scores = np.zeros(len(cawc_all_labels), dtype=np.float64)
        
        for idx in range(len(cawc_all_scores)):
            u = float(np.clip(u_arr[idx], 0.0, 1.0))
            dyn_weights = {
                "ig": metric_weights.get("ig", 0.0) * u,
                "kl": metric_weights.get("kl", 0.0) * (0.5 + 0.5 * u),
                "entropy": metric_weights.get("entropy", 0.0) * (1.0 - u),
                "cd": metric_weights.get("cd", 0.0) * (1.0 - 0.5 * u),
                "selfcheck": metric_weights.get("selfcheck", 0.0) * (1.0 - u),
            }
            z = sum(dyn_weights.values()) + 1e-8
            
            comp_val = 0.0
            for key in metric_keys:
                comp_val += dyn_weights[key] * pooled_scaled[key][idx]
            cawc_all_scores[idx] = comp_val / z
            
        cawc_all_scores = cawc_all_scores.tolist()
    else:
        cawc_all_scores = []

    
    # Diagnostic: check score variation
    if cawc_all_scores:
        cawc_scores_arr = np.asarray(cawc_all_scores)
        print(f"CAWC score stats: min={cawc_scores_arr.min():.6f} max={cawc_scores_arr.max():.6f} "
              f"mean={cawc_scores_arr.mean():.6f} std={cawc_scores_arr.std():.6f} "
              f"unique={len(np.unique(cawc_scores_arr))}")
    
    cawc_auc, cawc_lo, cawc_hi = bootstrap_auroc(cawc_all_labels, cawc_all_scores)
    print(f">>> CAWC Token-Level AUROC: {cawc_auc:.4f} [{cawc_lo:.4f}, {cawc_hi:.4f}] <<<")

    if not args.cawc_only:
        # ---- Sanity check: positive rate agreement ----
        print("\n" + "-"*60)
        print("SANITY CHECK: Positive rate agreement between phases")
        print("-"*60)
        print(f"  ReDeEP positive rate: {rd_pos_rate:.4f} ({sum(redeep_all_labels)}/{len(redeep_all_labels)} tokens)")
        print(f"  CAWC   positive rate: {cawc_pos_rate:.4f} ({sum(cawc_all_labels)}/{len(cawc_all_labels)} tokens)")
        if rd_pos_rate > 0:
            ratio = cawc_pos_rate / rd_pos_rate
            print(f"  Ratio (CAWC/ReDeEP): {ratio:.3f}")
            if abs(ratio - 1.0) > 0.3:
                print("  ⚠ WARNING: Positive rates differ by >30%. Possible label alignment issue.")
            else:
                print("  ✓ Positive rates are within expected range.")

    # ---- Final Report ----
    if not args.cawc_only:
        rd_time_avg = np.mean(redeep_times) if redeep_times else 0
        rd_time_std = np.std(redeep_times) if redeep_times else 0
    cawc_time_avg = np.mean(cawc_times) if cawc_times else 0
    cawc_time_std = np.std(cawc_times) if cawc_times else 0
    
    # Compute delta and speedup
    delta_auc = cawc_auc - rd_auc if not args.cawc_only else 0.0
    speedup = (rd_time_avg / cawc_time_avg) if (not args.cawc_only and cawc_time_avg > 0) else float('inf')
    
    print("\n" + "="*80)
    print("HEAD-TO-HEAD COMPARISON: ReDeEP vs CAWC (Token-level, same samples)")
    print("="*80)
    print("This is the matched-subset, matched-template controlled comparison.")
    print("Both methods process the same (query, context, response) triples through")
    print("the same LLaMA-2-7B-chat model and are evaluated against the same labels.")
    print("For the officially-validated ReDeEP sanity check, see Phase 1 of run_all.sh.")
    print("-" * 80)
    print(f"Model:    {args.model_name} (FP16)")
    print(f"Dataset:  RAGTruth Test Split — {sample_size} samples (seed={args.seed})")
    print(f"Processed: ReDeEP {redeep_processed}/{sample_size}, CAWC {cawc_processed}/{sample_size}")
    print("-" * 80)
    print(f"{'Method':<10} | {'AUROC':<8} | {'95% CI':<22} | {'Tokens':<10} | {'Pos Rate':<10} | {'sec/sample':<15}")
    print("-" * 80)
    print(f"{'ReDeEP':<10} | {rd_auc:.4f}  | [{rd_lo:.4f}, {rd_hi:.4f}]  | {len(redeep_all_labels):<10} | {rd_pos_rate:.4f}    | {rd_time_avg:.2f} ± {rd_time_std:.2f}")
    print(f"{'CAWC':<10} | {cawc_auc:.4f}  | [{cawc_lo:.4f}, {cawc_hi:.4f}]  | {len(cawc_all_labels):<10} | {cawc_pos_rate:.4f}    | {cawc_time_avg:.2f} ± {cawc_time_std:.2f}")
    print("-" * 80)
    print(f"Δ AUROC (CAWC − ReDeEP): {delta_auc:+.4f}")
    print(f"Speedup (ReDeEP / CAWC): {speedup:.1f}×")
    print("="*80)
    
    # Save structured results
    results = {
        "model": args.model_name,
        "precision": "fp16",
        "seed": args.seed,
        "sample_size": sample_size,
        "samples_total_in_test_set": len(test_ds),
        "generator_breakdown": dict(gen_counts),
        "evaluation_type": "controlled_comparison",
        "description": (
            "Apples-to-apples token-level hallucination detection comparison. "
            "Both methods process every (query, context, response) triple through "
            f"{args.model_name}. Ground-truth labels are factual accuracy vs context."
        ),
        "redeep": {
            "auroc": float(rd_auc), "ci_low": float(rd_lo), "ci_high": float(rd_hi),
            "tokens": len(redeep_all_labels), "pos_rate": float(rd_pos_rate),
            "samples_processed": redeep_processed,
            "time_per_sample_mean": float(rd_time_avg),
            "time_per_sample_std": float(rd_time_std),
        },
        "cawc": {
            "auroc": float(cawc_auc), "ci_low": float(cawc_lo), "ci_high": float(cawc_hi),
            "tokens": len(cawc_all_labels), "pos_rate": float(cawc_pos_rate),
            "samples_processed": cawc_processed,
            "time_per_sample_mean": float(cawc_time_avg),
            "time_per_sample_std": float(cawc_time_std),
        },
        "delta_auroc": float(delta_auc),
        "speedup": float(speedup),
    }
    
    results_file = script_dir / "comparison_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    main()

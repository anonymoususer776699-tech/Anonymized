import os
import json
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score

def safe_auroc(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    mask = np.isfinite(scores)
    l, s = labels[mask], scores[mask]
    if len(np.unique(l)) < 2:
        return 0.5
    return float(roc_auc_score(l, s))

def bootstrap_ci(labels, scores, n_boot=1000, seed=42):
    labels, scores = np.asarray(labels), np.asarray(scores)
    mask = np.isfinite(scores)
    l, s = labels[mask], scores[mask]
    if len(np.unique(l)) < 2:
        return 0.5, 0.5, 0.5
    
    base = roc_auc_score(l, s)
    rng = np.random.default_rng(seed)
    boots = []
    
    # Simple independent bootstrap for ReDeEP (since they flattened the DF)
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

def group_bootstrap_ci(groups, n_boot=1000, seed=42):
    """Bootstrap at the response level (like CAWC does)."""
    rng = np.random.default_rng(seed)
    boots = []
    n_groups = len(groups)
    
    for _ in range(n_boot):
        idx = rng.choice(n_groups, size=n_groups, replace=True)
        boot_l = []
        boot_s = []
        for i in idx:
            gy, gs = groups[i]
            mask = np.isfinite(gs)
            boot_l.extend(gy[mask])
            boot_s.extend(gs[mask])
            
        try:
            boots.append(roc_auc_score(boot_l, boot_s))
        except ValueError:
            pass
            
    # Base AUROC
    all_l, all_s = [], []
    for gy, gs in groups:
        mask = np.isfinite(gs)
        all_l.extend(gy[mask])
        all_s.extend(gs[mask])
        
    base = roc_auc_score(all_l, all_s)
    
    if not boots:
        return base, base, base
        
    return base, np.percentile(boots, 2.5), np.percentile(boots, 97.5)

def eval_pipeline():
    base_dir = os.path.dirname(__file__)
    
    print("=" * 80)
    print(" CAWC v5 vs ReDeEP Controlled Comparison (Llama-2-7B-base, fp16, matched subset)")
    print("=" * 80)
    
    # ---------------------------------------------------------
    # ReDeEP Evaluation
    # ---------------------------------------------------------
    redeep_csv = os.path.join(base_dir, "ReDEeP-ICLR", "ReDeEP", "log", "test_llama2_7B", "redeep_token_raw_scores.csv")
    redeep_time_file = os.path.join(base_dir, "ReDEeP-ICLR", "ReDeEP", "log", "test_llama2_7B", "redeep_wallclock.json")
    
    redeep_auroc = "N/A"
    redeep_ci = ("N/A", "N/A")
    redeep_response_auroc = "N/A"
    redeep_n = 0
    redeep_t_mean, redeep_t_std = 0.0, 0.0
    redeep_pos_rate = 0.0
    
    if os.path.exists(redeep_csv):
        df = pd.read_csv(redeep_csv)
        redeep_n = len(df)
        redeep_pos_rate = df['hallucination_label'].mean()
        
        # Token-level AUROC
        base, low, high = bootstrap_ci(df['hallucination_label'], df['difference_normalized'])
        redeep_auroc = f"{base:.4f} [{low:.4f}, {high:.4f}]"
        
        # Response-level AUROC (to match token_level_reg.py's printed 0.7331)
        df['response_group'] = df['identifier'].str.extract(r'(response_\d+)')
        grouped_df = df.groupby('response_group').agg(
            difference_normalized_mean=('difference_normalized', 'mean'),
            hallucination_label=('hallucination_label', 'max')
        ).reset_index()
        
        r_base, r_low, r_high = bootstrap_ci(grouped_df['hallucination_label'], grouped_df['difference_normalized_mean'])
        redeep_response_auroc = f"{r_base:.4f} [{r_low:.4f}, {r_high:.4f}]"
    else:
        print(f"[!] ReDeEP raw scores not found at {redeep_csv}")
        
    if os.path.exists(redeep_time_file):
        with open(redeep_time_file, "r") as f:
            times = json.load(f)
            if times:
                redeep_t_mean = np.mean(times)
                redeep_t_std = np.std(times)
                
    # ---------------------------------------------------------
    # CAWC Evaluation
    # ---------------------------------------------------------
    cawc_results_dir = os.environ.get("CAWC_RESULTS_DIR", "results_llama2_base")
    cawc_cache = os.path.join(base_dir, cawc_results_dir, "scores_cache_v5.pkl")
    cawc_time_file = os.path.join(base_dir, cawc_results_dir, "cawc_wallclock_ragtruth_test.json")
    
    cawc_auroc = "N/A"
    cawc_ci = ("N/A", "N/A")
    cawc_response_auroc = "N/A"
    cawc_n = 0
    cawc_t_mean, cawc_t_std = 0.0, 0.0
    cawc_pos_rate = 0.0
    
    if os.path.exists(cawc_cache):
        with open(cawc_cache, "rb") as f:
            data = pickle.load(f)
            results = data.get("results", [])
            
    elif os.path.exists(os.path.join(base_dir, "comparison_results.json")):
        with open(os.path.join(base_dir, "comparison_results.json"), "r") as f:
            comp_data = json.load(f)
            cawc_data = comp_data.get("cawc", {})
            cawc_auroc = f'{cawc_data.get("auroc", 0):.4f} [{cawc_data.get("ci_low", 0):.4f}, {cawc_data.get("ci_high", 0):.4f}]'
            cawc_n = cawc_data.get("tokens", 0)
            cawc_pos_rate = cawc_data.get("pos_rate", 0)
            cawc_t_mean = cawc_data.get("time_per_sample_mean", 0)
            cawc_t_std = cawc_data.get("time_per_sample_std", 0)
            
        results = []
            
        groups = []
        all_l = []
        cawc_response_groups = []
        for r in results:
            labels = np.array(r["labels"])
            comps = np.array(r["composite"])
            groups.append((labels, comps))
            all_l.extend(labels)
            cawc_n += len(labels)
            
            if len(labels) > 0:
                mask = np.isfinite(comps)
                if np.any(mask):
                    cawc_response_groups.append({
                        'label': np.max(labels),
                        'score': np.mean(comps[mask])
                    })
            
        if all_l:
            cawc_pos_rate = np.mean(all_l)
            
        if groups:
            base, low, high = group_bootstrap_ci(groups)
            cawc_auroc = f"{base:.4f} [{low:.4f}, {high:.4f}]"
            
        if cawc_response_groups:
            c_r_labels = [g['label'] for g in cawc_response_groups]
            c_r_scores = [g['score'] for g in cawc_response_groups]
            r_base, r_low, r_high = bootstrap_ci(c_r_labels, c_r_scores)
            cawc_response_auroc = f"{r_base:.4f} [{r_low:.4f}, {r_high:.4f}]"
    else:
        print(f"[!] CAWC cache not found at {cawc_cache}")
        
    if os.path.exists(cawc_time_file):
        with open(cawc_time_file, "r") as f:
            times = json.load(f)
            if times:
                cawc_t_mean = np.mean(times)
                cawc_t_std = np.std(times)
                
    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------
    print(f"\nMethod | Backbone | Precision | Tokens | Token AUROC | Response AUROC | sec/sample | Pos Rate")
    print("-" * 120)
    print(f"ReDeEP | Llama-2-7B | fp16 | {redeep_n} | {redeep_auroc} | {redeep_response_auroc} | {redeep_t_mean:.2f} ± {redeep_t_std:.2f} | {redeep_pos_rate:.1%}")
    print(f"CAWC v5 | Llama-2-7B | fp16 | {cawc_n} | {cawc_auroc} | {cawc_response_auroc} | {cawc_t_mean:.2f} ± {cawc_t_std:.2f} | {cawc_pos_rate:.1%}")
    print("\nNote: sec/sample represents wall-clock time per generator response.")

if __name__ == "__main__":
    eval_pipeline()

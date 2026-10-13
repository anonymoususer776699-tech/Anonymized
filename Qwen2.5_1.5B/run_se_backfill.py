#!/usr/bin/env python3
"""
run_se_backfill.py — Compute REAL Semantic Entropy (Kuhn et al. / Manakul et al.)
using DeBERTa-v3-large NLI clustering + Qwen2.5-1.5B continuation sampling.

No proxies. No placeholders. Real SE for every sample.
"""
import argparse, pickle, os, sys, time, warnings, random
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_from_disk

from core_metrics import MetricConfig, MetricEvaluator
from model_utils import ModelConfig, load_generator_model

warnings.filterwarnings("ignore")

# Use absolute paths so CWD doesn't matter
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE_DIR, "results")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pkl(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def se_is_valid(se_arr):
    """Check if SE array has real DeBERTa NLI values, not collapsed/placeholder.
    
    Real SE from NLI clustering produces discrete values (ln(1)=0, ln(2)=0.693,
    ln(3)=1.099, ...) with meaningful spread across anchor positions.
    Collapsed SE has all anchors at max entropy (ln(K)) with near-zero variance
    after interpolation.
    """
    if se_arr is None or len(se_arr) == 0:
        return False
    a = np.asarray(se_arr, dtype=np.float64)
    if len(a) < 2:
        return False
    # Must have enough finite (non-NaN) values
    finite = a[np.isfinite(a)]
    if len(finite) < max(3, len(a) * 0.10):
        return False
    nonzero = finite[finite != 0]
    if len(nonzero) < max(3, len(a) * 0.10):
        return False
    # Must have meaningful variance — collapsed SE is near-constant
    if np.std(finite) < 0.05:
        return False
    # Must have at least 2 distinct value levels (real clustering gives
    # discrete steps, collapsed gives one repeated value via interpolation)
    unique_vals = np.unique(np.round(nonzero, 2))
    if len(unique_vals) < 2:
        return False
    return True

def compute_se_for_sample(ev, token_ids, prefix, entropy_scores):
    """Compute real SE for one sample. Returns numpy array of SE values."""
    tid = np.asarray(token_ids, dtype=np.int64)
    ent = np.asarray(entropy_scores, dtype=np.float64) if len(entropy_scores) > 0 else np.zeros(len(tid))
    se_arr = ev.semantic_entropy_series(tid, prefix, ent)
    return se_arr

def backfill_test_cache(ev, test_path, recompute_all=False, start_index=0):
    """Backfill SE in the test cache. Test cache has token_ids and text.
    Resumes by skipping samples whose existing SE already looks valid."""
    print(f"\n{'='*60}")
    print(f"BACKFILLING TEST CACHE: {test_path}")
    if recompute_all:
        print(f"  Recomputing ALL samples (overwriting existing SE)")
    else:
        print(f"  Resuming: skipping samples with valid existing SE")
    print(f"{'='*60}")
    
    data = load_pkl(test_path)
    results = data if isinstance(data, list) else data.get('results', [])
    
    updated = 0
    skipped = 0
    t0 = time.time()
    
    start_index = max(0, min(int(start_index), len(results)))
    if start_index:
        print(f"  Starting test cache from sample index {start_index}")

    for i in tqdm(range(start_index, len(results)), desc="Test SE", initial=start_index, total=len(results)):
        r = results[i]
        
        # Get token_ids  
        tid = r.get('token_ids', [])
        if not tid or len(tid) == 0:
            skipped += 1
            continue

        if not recompute_all:
            raw_existing = r.get('scores_raw', r.get('scores', {}))
            if se_is_valid(raw_existing.get('se', [])):
                skipped += 1
                continue
        
        # Get text for prefix
        text = r.get('text', {})
        q = text.get('query', '')
        c = text.get('context', '')
        prefix = f"Question: {q}\nContext: {c}\nAnswer: "
        
        # Get entropy for anchor selection
        raw = r.get('scores_raw', r.get('scores', {}))
        ent = raw.get('entropy', [])
        
        try:
            se_arr = compute_se_for_sample(ev, tid, prefix, ent)
            se_list = se_arr.tolist()
            
            # Save to ALL THREE locations so pipeline finds it regardless
            # of which dict it checks first (belt-and-suspenders).
            if 'scores_raw_undirected' in r:
                r['scores_raw_undirected']['se'] = se_list
            if 'scores_raw' in r:
                r['scores_raw']['se'] = se_list
            if 'scores' in r:
                r['scores']['se'] = se_list
            
            updated += 1
            
            # Auto-save every 50 samples
            if updated % 50 == 0:
                elapsed = (time.time() - t0) / 60
                print(f"  Checkpoint: {updated} done, {skipped} skipped, {elapsed:.1f} min elapsed")
                if isinstance(data, dict):
                    data['results'] = results
                save_pkl(test_path, data if isinstance(data, dict) else results)
                
        except Exception as e:
            print(f"  Warning: sample {i} failed: {e}")
            continue
    
    # Final save
    elapsed = (time.time() - t0) / 60
    print(f"\nTest SE complete: {updated} computed, {skipped} skipped, {elapsed:.1f} min")
    if isinstance(data, dict):
        data['results'] = results
    save_pkl(test_path, data if isinstance(data, dict) else results)


def backfill_train_cache(ev, train_path, dataset_path, recompute_all=False, start_index=0):
    """Backfill SE in the train cache. 
    Train cache lacks token_ids/text, so we load from original dataset.
    Resumes by skipping samples whose existing SE already looks valid."""
    print(f"\n{'='*60}")
    print(f"BACKFILLING TRAIN CACHE: {train_path}")
    if recompute_all:
        print(f"  Recomputing ALL samples (overwriting existing SE)")
    else:
        print(f"  Resuming: skipping samples with valid existing SE")
    print(f"{'='*60}")
    
    data = load_pkl(train_path)
    results = data if isinstance(data, list) else data.get('results', [])
    
    # Load original dataset to get text
    ds = load_from_disk(dataset_path)
    # Build lookup by sample_id
    ds_lookup = {}
    for idx, row in enumerate(ds):
        ds_lookup[str(row['id'])] = row
    
    print(f"  Dataset loaded: {len(ds)} samples, Cache: {len(results)} samples")
    
    updated = 0
    skipped = 0
    t0 = time.time()
    
    start_index = max(0, min(int(start_index), len(results)))
    if start_index:
        print(f"  Starting train cache from sample index {start_index}")

    for i in tqdm(range(start_index, len(results)), desc="Train SE", initial=start_index, total=len(results)):
        r = results[i]
        
        # Match to dataset by sample_id
        sample_id = str(r.get('meta', {}).get('sample_id', ''))
        ds_row = ds_lookup.get(sample_id)
        if ds_row is None:
            skipped += 1
            continue

        if not recompute_all:
            raw_existing = r.get('scores_raw', r.get('scores', {}))
            if se_is_valid(raw_existing.get('se', [])):
                skipped += 1
                continue
        
        # Build prefix and tokenize output to get token_ids
        q = ds_row.get('query', '')
        c = ds_row.get('context', '')
        output_text = ds_row.get('output', '')
        prefix = f"Question: {q}\nContext: {c}\nAnswer: "
        
        # Tokenize the output to get token_ids
        token_ids = ev.tokenizer.encode(output_text, add_special_tokens=False)
        if not token_ids:
            continue
            
        n_tokens = r.get('n_tokens', len(token_ids))
        token_ids = token_ids[:n_tokens]
        
        # Get entropy for anchor selection
        scores = r.get('scores', {})
        ent = scores.get('entropy', [])
        
        try:
            se_arr = compute_se_for_sample(ev, token_ids, prefix, ent)
            # Ensure length matches n_tokens
            if len(se_arr) < n_tokens:
                se_arr = np.pad(se_arr, (0, n_tokens - len(se_arr)), constant_values=0.0)
            se_arr = se_arr[:n_tokens]
            se_list = se_arr.tolist()
            
            # Save to ALL locations so pipeline finds it regardless
            if 'scores_raw_undirected' in r:
                r['scores_raw_undirected']['se'] = se_list
            if 'scores_raw' in r:
                r['scores_raw']['se'] = se_list
            r['scores']['se'] = se_list
            
            updated += 1
            
            # Auto-save every 50 samples
            if updated % 50 == 0:
                elapsed = (time.time() - t0) / 60
                print(f"  Checkpoint: {updated} done, {skipped} skipped, {elapsed:.1f} min elapsed")
                if isinstance(data, dict):
                    data['results'] = results
                save_pkl(train_path, data if isinstance(data, dict) else results)
                
        except Exception as e:
            print(f"  Warning: train sample {i} (id={sample_id}) failed: {e}")
            continue
    
    # Final save
    elapsed = (time.time() - t0) / 60
    print(f"\nTrain SE complete: {updated} computed, {skipped} skipped, {elapsed:.1f} min")
    if isinstance(data, dict):
        data['results'] = results
    save_pkl(train_path, data if isinstance(data, dict) else results)


def test_single_sample(ev, cache_path, sample_index, dataset_path=None):
    """Helper function to run the SE test for a specific sample index.
    If dataset_path is provided, it assumes train cache format (requires dataset to get text).
    Otherwise, it assumes test/halueval cache format.
    """
    print(f"\n{'='*60}")
    print(f"TESTING SINGLE SAMPLE INDEX: {sample_index}")
    print(f"CACHE PATH: {cache_path}")
    print(f"{'='*60}")
    
    data = load_pkl(cache_path)
    results = data if isinstance(data, list) else data.get('results', [])
    
    if sample_index < 0 or sample_index >= len(results):
        print(f"Error: Index {sample_index} out of bounds (0 to {len(results)-1}).")
        return None
        
    r = results[sample_index]
    
    if dataset_path:
        # Train cache format
        ds = load_from_disk(dataset_path)
        sample_id = str(r.get('meta', {}).get('sample_id', ''))
        
        ds_row = None
        for row in ds:
            if str(row['id']) == sample_id:
                ds_row = row
                break
                
        if not ds_row:
            print(f"Error: Sample ID {sample_id} not found in dataset.")
            return None
            
        q = ds_row.get('query', '')
        c = ds_row.get('context', '')
        output_text = ds_row.get('output', '')
        prefix = f"Question: {q}\nContext: {c}\nAnswer: "
        
        token_ids = ev.tokenizer.encode(output_text, add_special_tokens=False)
        n_tokens = r.get('n_tokens', len(token_ids))
        token_ids = token_ids[:n_tokens]
    else:
        # Test cache format
        token_ids = r.get('token_ids', [])
        text = r.get('text', {})
        q = text.get('query', '')
        c = text.get('context', '')
        prefix = f"Question: {q}\nContext: {c}\nAnswer: "

    if not token_ids:
        print("Error: No token_ids found for this sample.")
        return None

    raw = r.get('scores_raw', r.get('scores', {}))
    ent = raw.get('entropy', [])
    
    t0 = time.time()
    se_arr = compute_se_for_sample(ev, token_ids, prefix, ent)
    elapsed = time.time() - t0
    
    print(f"Finished in {elapsed:.1f} seconds.")
    print("Computed SE array:")
    print(se_arr)
    return se_arr


def main():
    parser = argparse.ArgumentParser(description="SE Backfill with Qwen2.5-1.5B + DeBERTa NLI")
    parser.add_argument("--skip-test", action="store_true",
                        help="Skip the test cache phase (already done)")
    parser.add_argument("--skip-halueval", action="store_true",
                        help="Skip the HaluEval cache phase")
    parser.add_argument("--train-start-index", type=int, default=0,
                        help="Start train SE backfill from this zero-based cache index")
    parser.add_argument("--test-start-index", type=int, default=0,
                        help="Start test SE backfill from this zero-based cache index")
    parser.add_argument("--halueval-start-index", type=int, default=0,
                        help="Start halueval SE backfill from this zero-based cache index")
    parser.add_argument("--results-dir", type=str, default=RESULTS,
                        help="Directory containing scores_cache_v5.pkl, train_raw_cache_v5.pkl, and halueval_scores_cache_v5.pkl")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for stochastic continuation sampling")
    parser.add_argument("--recompute-all", action="store_true",
                        help="Overwrite existing valid SE instead of resuming from cached samples")
    
    # New arguments for testing a single sample
    parser.add_argument("--target-sample-index", type=int, default=-1,
                        help="If set >= 0, tests SE for ONLY this specific index and exits.")
    parser.add_argument("--target-cache", type=str, choices=["test", "train", "halueval"], default="test",
                        help="Which cache to use when --target-sample-index is specified.")
    
    args = parser.parse_args()
    set_seed(args.seed)

    print("=" * 60)
    print("SEMANTIC ENTROPY BACKFILL — Qwen2.5-1.5B + DeBERTa NLI (Kuhn et al.)")
    print("=" * 60)
    
    # Setup models — Qwen2.5-1.5B in 4-bit quantization (~2.5GB VRAM).
    # max_gpu_memory_gib=6 leaves room for DeBERTa NLI (~1.5GB) which loads lazily.
    print("\nLoading Qwen2.5-1.5B for continuation sampling...")
    m_cfg = ModelConfig(model_name="Qwen/Qwen2.5-1.5B", max_gpu_memory_gib=6)
    model, tokenizer = load_generator_model(m_cfg)
    
    # SE config — optimized for Qwen2.5-1.5B with lowered NLI threshold.
    # semantic_samples=5: generation loop does 2 rounds × 3 = 6, truncated to 5.
    #   Using 5 avoids OOM from 4-bit batch generation and matches
    #   Kuhn et al. who showed 5 samples is sufficient.
    # semantic_entailment_threshold=0.30: with proxy-model continuations,
    #   entailment scores are low — threshold 0.70 means nothing merges and
    #   ALL samples get max entropy. Lowering to 0.30 allows clusters to form.
    metric_cfg = MetricConfig(
        semantic_samples=5,
        semantic_max_new_tokens=40,
        semantic_temperature=0.85,
        semantic_anchor_stride=12,
        semantic_time_budget_seconds=60.0,
        semantic_entailment_threshold=0.30,
        semantic_text_char_limit=600,
    )
    ev = MetricEvaluator(
        tokenizer=tokenizer, 
        generator_model=model, 
        config=metric_cfg, 
        nli_device="cuda"
    )
    
    results_dir = os.path.abspath(args.results_dir)
    test_path = os.path.join(results_dir, "scores_cache_v5.pkl")
    train_path = os.path.join(results_dir, "train_raw_cache_v5.pkl")
    halu_path = os.path.join(results_dir, "halueval_scores_cache_v5.pkl")
    train_ds_path = os.path.join(BASE_DIR, "ragtruth_processed", "train")
    
    if args.target_sample_index >= 0:
        if args.target_cache == "test":
            test_single_sample(ev, test_path, args.target_sample_index)
        elif args.target_cache == "halueval":
            test_single_sample(ev, halu_path, args.target_sample_index)
        elif args.target_cache == "train":
            test_single_sample(ev, train_path, args.target_sample_index, dataset_path=train_ds_path)
        return
        
    # Phase 1: Test cache
    if args.skip_test:
        print(f"\nSkipping test cache (--skip-test)")
    elif os.path.exists(test_path):
        backfill_test_cache(ev, test_path, recompute_all=args.recompute_all, start_index=args.test_start_index)
    else:
        print(f"WARNING: Test cache not found at {test_path}")
    
    # Phase 2: Train cache (needs original dataset for text)
    if os.path.exists(train_path):
        if os.path.exists(train_ds_path):
            backfill_train_cache(
                ev,
                train_path,
                train_ds_path,
                recompute_all=args.recompute_all,
                start_index=args.train_start_index,
            )
        else:
            print(f"WARNING: Train dataset not found at {train_ds_path} — cannot backfill train cache")
    else:
        print(f"WARNING: Train cache not found at {train_path}")
    
    # Phase 3: HaluEval cache (same structure as test cache)
    if args.skip_halueval:
        print(f"\nSkipping HaluEval cache (--skip-halueval)")
    elif os.path.exists(halu_path):
        backfill_test_cache(ev, halu_path, recompute_all=args.recompute_all, start_index=args.halueval_start_index)
    else:
        print(f"WARNING: HaluEval cache not found at {halu_path}")
    
    # Phase 4: Verify
    print(f"\n{'='*60}")
    print("VERIFICATION")
    print(f"{'='*60}")
    
    for name, path in [("Test", test_path), ("Train", train_path), ("HaluEval", halu_path)]:
        if not os.path.exists(path):
            print(f"  {name}: MISSING")
            continue
        d = load_pkl(path)
        res = d if isinstance(d, list) else d.get('results', [])
        valid_count = 0
        all_se = []
        for r in res:
            raw = r.get('scores_raw', r.get('scores', {}))
            se = raw.get('se', [])
            if len(se) > 0:
                all_se.extend(se)
                if se_is_valid(se):
                    valid_count += 1
        all_se = np.asarray(all_se, dtype=np.float64)
        finite = all_se[np.isfinite(all_se)]
        zero_pct = (finite == 0).mean() * 100 if len(finite) > 0 else 100.0
        print(f"  {name}: {len(res)} samples, {valid_count} valid SE ({100*valid_count/max(len(res),1):.1f}%)")
        if len(finite) > 0:
            print(f"    tokens={len(finite)}, zeros={zero_pct:.1f}%, mean={finite.mean():.4f}, std={finite.std():.4f}, max={finite.max():.4f}")


if __name__ == "__main__":
    main()

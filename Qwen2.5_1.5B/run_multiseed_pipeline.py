#!/usr/bin/env python3
"""Run the full CAWC pipeline across multiple seeds and aggregate AUROC."""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev


BASE_DIR = Path(__file__).resolve().parent
CACHE_FILES = [
    "train_raw_cache_v5.pkl",
    "scores_cache_v5.pkl",
    "halueval_scores_cache_v5.pkl",
]


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def load_results(cache_path: Path) -> list[dict]:
    if not cache_path.exists():
        return []
    with cache_path.open("rb") as f:
        payload = pickle.load(f)
    return payload if isinstance(payload, list) else payload.get("results", [])


def load_payload(cache_path: Path):
    if not cache_path.exists():
        return None
    with cache_path.open("rb") as f:
        return pickle.load(f)


def has_real_selfcheck_marker(cache_path: Path, split: str) -> bool:
    payload = load_payload(cache_path)
    if not isinstance(payload, dict):
        return False
    meta = payload.get("backfill_metadata", {})
    return split in set(meta.get("selfcheck_nli_splits", []))


def has_metric(entry: dict, metric: str) -> bool:
    for bucket in ("scores_raw_undirected", "scores_raw", "scores"):
        arr = entry.get(bucket, {}).get(metric, [])
        if arr:
            return True
    return False


def verify_seed_dir(results_dir: Path, strict: bool = True) -> dict:
    report = {}
    expected = {
        "train": results_dir / "train_raw_cache_v5.pkl",
        "test": results_dir / "scores_cache_v5.pkl",
        "halueval": results_dir / "halueval_scores_cache_v5.pkl",
    }
    for split, path in expected.items():
        rows = load_results(path)
        se_count = sum(1 for row in rows if has_metric(row, "se"))
        sc_count = sum(1 for row in rows if has_metric(row, "selfcheck"))
        report[split] = {
            "samples": len(rows),
            "se_samples": se_count,
            "selfcheck_samples": sc_count,
            "selfcheck_nli_marked": has_real_selfcheck_marker(path, split),
        }
        if strict:
            if not rows:
                raise RuntimeError(f"{split} cache is missing or empty in {results_dir}")
            if se_count != len(rows):
                raise RuntimeError(f"{split} SE incomplete in {results_dir}: {se_count}/{len(rows)}")
            if sc_count != len(rows):
                raise RuntimeError(f"{split} SelfCheck incomplete in {results_dir}: {sc_count}/{len(rows)}")
            if split == "halueval" and not has_real_selfcheck_marker(path, split):
                raise RuntimeError(f"{split} SelfCheck-NLI marker missing in {results_dir}; proxy scores may still be present")
    return report


def adopt_current_results(seed_dir: Path, source_dir: Path, overwrite: bool) -> None:
    seed_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for src in source_dir.iterdir():
        if not src.is_file():
            continue
        if src.name.startswith("multiseed_"):
            continue
        dst = seed_dir / src.name
        if dst.exists() and not overwrite:
            continue
        shutil.copy2(src, dst)
        copied.append(src.name)
    for name in CACHE_FILES:
        if not (seed_dir / name).exists():
            raise FileNotFoundError(f"Cannot adopt current run; missing {source_dir / name}")
    metadata = {"adopted_from": str(source_dir), "seed": 42, "files_copied": copied}
    (seed_dir / "seed_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def caches_exist(seed_dir: Path) -> bool:
    return all((seed_dir / name).exists() for name in CACHE_FILES)


def run_seed(seed: int, args: argparse.Namespace) -> Path:
    seed_dir = BASE_DIR / args.results_root / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    rel_seed_dir = os.path.relpath(seed_dir, BASE_DIR)

    if not caches_exist(seed_dir) or args.force_recompute:
        run_cmd([
            sys.executable,
            "run_pipeline_v2.py",
            "--base-dir", ".",
            "--results-dir", rel_seed_dir,
            "--seed", str(seed),
            "--force-recompute",
            "--cache-every", str(args.cache_every),
        ], BASE_DIR)

    skip_existing_backfills = False
    if args.trust_existing_complete and not args.force_recompute and caches_exist(seed_dir):
        try:
            verify_seed_dir(seed_dir, strict=True)
            skip_existing_backfills = True
            print(f"Seed {seed} already has complete train/test SE and SelfCheck; skipping backfills.")
        except RuntimeError:
            skip_existing_backfills = False

    if args.run_backfills and not skip_existing_backfills:
        run_cmd([
            sys.executable,
            "run_se_backfill.py",
            "--results-dir", str(seed_dir),
            "--seed", str(seed),
        ], BASE_DIR)
        run_cmd([
            sys.executable,
            "run_selfcheck_backfill.py",
            "--results-dir", str(seed_dir),
            "--seed", str(seed),
        ], BASE_DIR)

    report = verify_seed_dir(seed_dir, strict=not args.allow_incomplete)
    (seed_dir / "cache_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    run_cmd([
        sys.executable,
        "run_final_unsupervised.py",
        "--results-dir", str(seed_dir),
        "--bootstrap-seed", str(seed),
    ], BASE_DIR)
    return seed_dir


def read_seed_aurocs(seed: int, seed_dir: Path) -> list[dict]:
    table = seed_dir / "experiment_1_2_table.csv"
    if not table.exists():
        raise FileNotFoundError(f"Missing final AUROC table: {table}")
    with table.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        out.append({
            "Seed": seed,
            "Method": row["Method"],
            "AUROC": float(row["AUROC"]),
        })
    return out


def write_aggregate(seed_rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_method: dict[str, list[float]] = {}
    for row in seed_rows:
        by_method.setdefault(row["Method"], []).append(row["AUROC"])

    summary_rows = []
    for method, values in by_method.items():
        summary_rows.append({
            "Method": method,
            "N_seeds": len(values),
            "AUROC_mean": f"{mean(values):.4f}",
            "AUROC_std": f"{(stdev(values) if len(values) > 1 else 0.0):.4f}",
            "AUROC_mean_std": f"{mean(values):.4f} +/- {(stdev(values) if len(values) > 1 else 0.0):.4f}",
        })

    with (out_dir / "multiseed_auroc_by_seed.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Seed", "Method", "AUROC"])
        writer.writeheader()
        writer.writerows(seed_rows)

    with (out_dir / "multiseed_auroc_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Method", "N_seeds", "AUROC_mean", "AUROC_std", "AUROC_mean_std"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nMean +/- std AUROC")
    for row in summary_rows:
        print(f"  {row['Method']}: {row['AUROC_mean_std']} (n={row['N_seeds']})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and aggregate CAWC pipeline across seeds.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[100],
                        help="Seeds to run. Use at least three for headline reporting.")
    parser.add_argument("--results-root", type=str, default="results",
                        help="Root directory for seed_<n> result directories")
    parser.add_argument("--adopt-current-as-42", action="store_true",
                        help="Copy top-level results caches into results/seed_42 and count as seed 42")
    parser.add_argument("--overwrite-adopted", action="store_true",
                        help="Overwrite existing results/seed_42 caches during adoption")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Re-run run_pipeline_v2.py even when seed caches exist")
    parser.add_argument("--skip-backfills", dest="run_backfills", action="store_false",
                        help="Skip SE/SelfCheck backfills, useful only if already complete")
    parser.set_defaults(run_backfills=True)
    parser.add_argument("--trust-existing-complete", action="store_true",
                        help="Skip backfills for seed directories whose train/test caches already contain SE and SelfCheck")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Do not fail when train/test SE or SelfCheck coverage is incomplete")
    parser.add_argument("--cache-every", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(set(args.seeds)) < 3:
        print("Note: Running with fewer than 3 seeds. Mean +/- std reporting will be limited.")

    results_root = BASE_DIR / args.results_root
    if args.adopt_current_as_42:
        adopt_current_results(results_root / "seed_42", BASE_DIR / "results", args.overwrite_adopted)

    seed_rows = []
    for seed in args.seeds:
        if args.adopt_current_as_42 and seed == 42:
            seed_dir = results_root / "seed_42"
            report = verify_seed_dir(seed_dir, strict=False)
            (seed_dir / "cache_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print("Seed 42 adopted from current results; skipping pipeline and backfills.")
        else:
            seed_dir = run_seed(seed, args)
        seed_rows.extend(read_seed_aurocs(seed, seed_dir))

    write_aggregate(seed_rows, results_root)


if __name__ == "__main__":
    main()

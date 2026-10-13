#!/bin/bash
set -euo pipefail

# Resolve absolute directory of this script, regardless of where it's invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[$(date)] Script directory: $SCRIPT_DIR"

# Enable conda
CONDA_BASE=$(conda info --base)
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Override these from the shell if the remote machine needs a different setup.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MODEL_NAME="${MODEL_NAME:-NousResearch/Llama-2-7b-hf}"
REDEEP_MODEL_NAME="${REDEEP_MODEL_NAME:-llama2-7b}"
RESULTS_DIR="${RESULTS_DIR:-results_llama2_base}"
QUANTIZE="${QUANTIZE:-0}"
FORCE_RECOMPUTE="${FORCE_RECOMPUTE:-0}"

if [ "$QUANTIZE" = "1" ]; then
    CAWC_QUANTIZE_ARG=""
else
    CAWC_QUANTIZE_ARG="--no-quantize"
fi
if [ "$FORCE_RECOMPUTE" = "1" ]; then
    CAWC_FORCE_ARG="--force-recompute"
else
    CAWC_FORCE_ARG=""
fi

echo "[$(date)] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[$(date)] MODEL_NAME=$MODEL_NAME"
echo "[$(date)] RESULTS_DIR=$RESULTS_DIR"
echo "[$(date)] QUANTIZE=$QUANTIZE"
echo "[$(date)] FORCE_RECOMPUTE=$FORCE_RECOMPUTE"

# ==========================================
# Environment Setup: ReDeEP (custom transformers fork)
# ==========================================
if ! conda env list | grep -q "env_redeep"; then
    echo "[$(date)] Creating Conda env_redeep (Python 3.10)..."
    conda create -n env_redeep python=3.10 -y
fi
echo "[$(date)] Installing ReDeEP dependencies..."
conda activate env_redeep
cd "$SCRIPT_DIR/ReDEeP-ICLR/transformers"
pip install huggingface_hub accelerate
pip install -e .
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install tqdm scikit-learn scipy pandas datasets matplotlib sentence-transformers
pip install bitsandbytes==0.43.1
python - <<'PY'
import importlib.metadata as md
print("bitsandbytes", md.version("bitsandbytes"))
PY
conda deactivate

# ==========================================
# Phase 1: ReDeEP standalone (full 2700 test set, for the official score)
# ==========================================
echo "[$(date)] Starting ReDeEP Phase (full test set)..."

REDEEP_PYTHON="$CONDA_BASE/envs/env_redeep/bin/python"
cd "$SCRIPT_DIR/ReDEeP-ICLR/ReDeEP"
mkdir -p log/test_llama2_7B

echo "[$(date)] Running ReDeEP token_level_detect..."
$REDEEP_PYTHON token_level_detect.py --model_name "$REDEEP_MODEL_NAME" --dataset ragtruth --model_path "$MODEL_NAME"
RD_DETECT_EXIT=$?
echo "[$(date)] token_level_detect exit code: $RD_DETECT_EXIT"

if [ $RD_DETECT_EXIT -ne 0 ]; then
    echo "[$(date)] ERROR: token_level_detect failed. Aborting."
    exit 1
fi

echo "[$(date)] Running ReDeEP token_level_reg..."
$REDEEP_PYTHON token_level_reg.py --model_name "$REDEEP_MODEL_NAME" --dataset ragtruth
RD_REG_EXIT=$?
echo "[$(date)] token_level_reg exit code: $RD_REG_EXIT"

if [ $RD_REG_EXIT -ne 0 ]; then
    echo "[$(date)] ERROR: token_level_reg failed. Aborting."
    exit 1
fi

# ==========================================
# Phase 2: Full CAWC Pipeline (CAWC)
# ==========================================
echo "[$(date)] Starting full CAWC pipeline..."
cd "$SCRIPT_DIR"

PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}" $REDEEP_PYTHON run_pipeline_v2.py \
    --model-name "$MODEL_NAME" \
    $CAWC_QUANTIZE_ARG \
    $CAWC_FORCE_ARG \
    --seed 42 \
    --results-dir "$RESULTS_DIR"
COMPARE_EXIT=$?
echo "[$(date)] Pipeline exit code: $COMPARE_EXIT"

if [ $COMPARE_EXIT -ne 0 ]; then
    echo "[$(date)] ERROR: run_pipeline_v2.py failed."
    exit 1
fi

# ==========================================
# Phase 3: CAWC Experiment Tables (Experiment 1/2 and companion analyses)
# ==========================================
echo "[$(date)] Running CAWC final unsupervised evaluation tables..."
cd "$SCRIPT_DIR"
PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}" $REDEEP_PYTHON run_final_unsupervised.py --results-dir "$RESULTS_DIR"
FINAL_EVAL_EXIT=$?
echo "[$(date)] CAWC final eval exit code: $FINAL_EVAL_EXIT"

if [ $FINAL_EVAL_EXIT -ne 0 ]; then
    echo "[$(date)] ERROR: run_final_unsupervised.py failed."
    exit 1
fi

# ==========================================
# Phase 4: Unified Evaluation (reads ReDeEP CSV + CAWC results/timing)
# ==========================================
echo "[$(date)] Running Unified Evaluation..."
cd "$SCRIPT_DIR"
CAWC_RESULTS_DIR="$RESULTS_DIR" PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}" $REDEEP_PYTHON step4_unified_eval.py
EVAL_EXIT=$?
echo "[$(date)] Unified eval exit code: $EVAL_EXIT"

echo "[$(date)] Pipeline finished. Exit codes: detect=$RD_DETECT_EXIT reg=$RD_REG_EXIT compare=$COMPARE_EXIT final_eval=$FINAL_EVAL_EXIT unified_eval=$EVAL_EXIT"

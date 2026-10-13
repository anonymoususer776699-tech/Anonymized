# Llama 2 7B Remote Run

Copy this directory to the GPU machine. Before running, supply the external `ReDEeP-ICLR` source tree (including its `transformers` fork and `ReDeEP` scripts), the processed datasets, and Conda. These prerequisites are not bundled in this workspace. Then run from inside the copied directory:

```bash
chmod +x run_all.sh
./run_all.sh
```

The default run uses:

- ReDeEP model key: `llama2-7b`
- Backbone path: `NousResearch/Llama-2-7b-hf`
- CAWC precision: fp16 via `--no-quantize`
- Output folder: `results_llama2_base`
- GPU: `CUDA_VISIBLE_DEVICES=0`

Key CAWC outputs in `results_llama2_base`:

- `scores_cache_v5.pkl`
- `train_raw_cache_v5.pkl`
- `halueval_scores_cache_v5.pkl`
- `experiment_1_2_table.csv`
- `experiment_1_2_table.txt`
- `final_unsupervised_log.txt`
- `cawc_wallclock_ragtruth_test.json`
- `pipeline_run_summary_v4.json`

The unified ReDeEP comparison reads `cawc_wallclock_ragtruth_test.json` for CAWC sec/sample.

You can override the defaults without editing files:

```bash
CUDA_VISIBLE_DEVICES=1 \
MODEL_NAME=NousResearch/Llama-2-7b-hf \
RESULTS_DIR=results_llama2_base \
./run_all.sh
```

For 4-bit CAWC, use:

```bash
QUANTIZE=1 ./run_all.sh
```

If a run failed and left a lock file, remove only the CAWC lock before restarting:

```bash
rm -f results_llama2_base/pipeline_v4.lock
```

If an old partial CAWC cache exists and you want fresh timing for every sample, run:

```bash
FORCE_RECOMPUTE=1 ./run_all.sh
```

The setup step installs `bitsandbytes==0.43.1` and verifies that Python can see its package metadata. That fixes the previous error:

```text
importlib.metadata.PackageNotFoundError: No package metadata was found for bitsandbytes
```

If you switch `MODEL_NAME` to a gated Meta Llama model, log in on the remote first:

```bash
huggingface-cli login
```

Then run:

```bash
MODEL_NAME=meta-llama/Llama-2-7b-hf ./run_all.sh
```

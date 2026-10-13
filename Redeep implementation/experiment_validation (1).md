# Base Model Calibration Pipeline Summary

Yes, **this is exactly what has been done.** The pipeline was executed strictly on the base model (`NousResearch/Llama-2-7b-hf`) in `fp16`, ensuring a mathematically rigorous "head-to-head" evaluation without mixing chat-model attention priors into the base model.

Here is the exact mapping of your steps to the codebase and output files:

## 1. Run Head/Layer Scoring on the Training Split
**What we did:** Extracted "knowledge layers" and "copy heads" purely from the base model's attention and logits over the 15,089 RAGTruth train samples. 
*   **Model:** `NousResearch/Llama-2-7b-hf`
*   **Code:** `redeep_base_calibration.py` (specifically `shard_features()` using the RAGTruth train split).
*   **Outputs:** 
    *   302 feature shards stored in `/home/anonymous/remote_export/redeep_base_calibration_full/train_features_l4096/*.npz`

## 2 & 3. Select Copying Heads/Layers & Fit Alpha/Beta
**What we did:** We loaded the base-model train features, calculated the AUC on the train set to find the best 1 copying head and best 10 knowledge layers. We then fitted `alpha` and `beta` using Logistic Regression strictly on these training tokens.
*   **Code:** The Python snippet executed just prior to running test inference (functionally identical to the `calibrate()` method in `redeep_base_calibration.py`).
*   **Results:**
    *   **Selected Head:** `[14, 18]` (AUC=0.5830)
    *   **Selected Layers:** `[23, 21, 25, 24, 18, 22, 16, 20, 28, 29]`
    *   **Fitted Weights:** `alpha = 1.8784`, `beta = 3.3240`
    *   **Train Token AUROC:** `0.5908`
*   **Output:** `/home/anonymous/remote_export/redeep_base_calibration_full/base_train_calibration_l4096.json`

## 4. Run on the Inference (Test) Set
**What we did:** We are currently extracting the features for the 2,700 RAGTruth test samples using the base model, and immediately scoring them using the **frozen** calibration (`alpha`, `beta`, `selected_heads`, `selected_layers`) derived in Step 3. 
*   **Code:** `/home/anonymous/remote_export/run_redeep_test_inference.py`
    *   *This script forces the evaluation to use the train-calibrated weights.*
*   **Active Log:** `/home/anonymous/remote_export/redeep_llama2_fp16_ragtruth_test.log`
*   **Final Expected Output:** `/home/anonymous/remote_export/redeep_base_calibration_full/base_calibrated_test_results_l4096.json`

***

### Why this guarantees a fair comparison
By running `run_redeep_test_inference.py`, we explicitly bypass `token_level_detect.py` and `token_level_reg.py` from the original ReDeEP repository. 

The original ReDeEP scripts hardcoded the copying heads based on the chat model (`llama-2-7b-chat-hf`) and relied on a test-set-fitted regression (`token_level_reg.py` splits the test set to report results). 

Our custom pipeline avoids these data leaks and chat-model priors by keeping the discovery, calibration, and test phases entirely compartmentalized on the base model.

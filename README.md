# SENTINEL

This document presents the summarized performance of the SENTINEL defense against multiple jailbreak attacks on different models. The table below displays the Benign FPR and attack success rates (ASR) for various attack methods when using SENTINEL as a defense.

### SENTINEL Defense Results Across Models

| Model         | Benign FPR ↓ | OR-Bench FPR ↓ | GCG ↓ | PAIR ↓ | GPT-Fuzz ↓ | FewShot ↓ | AutoDAN ↓ | RADICAL ↓ |
|---------------|--------------|----------------|--------|---------|--------------|------------|-------------|--------------|
| Llama2-7b     | 3.05         | 34.02          | 5.66   | 2.33    | 1.00         | 4.33       | 0.00        | 8.33         |
| Llama3-8b     | 3.09         | 29.70          | 0.67   | 1.66    | 1.67         | 3.00       | 0.33        | 6.33         |
| Mistral-7b-v2 | 1.59         | 24.66          | 4.00   | 2.00    | 2.67         | 4.67       | 0.67        | 9.33         |
| Vicuna-7b     | 2.05         | 30.76          | 6.00   | 7.33    | 4.33         | 4.33       | 2.67        | 7.00         |

## Description
- **Benign FPR**: False positive rate on benign/boundary inputs.
- Lower values in Benign FPR indicate less over-refusal.
- **Attack Columns**: Each attack method (GCG, PAIR, GPT-Fuzz, FewShot, AutoDAN) has two metrics—one for "Refusal Keywords" and one for Harmbench Classifier.
- Lower values in the attack columns indicate stronger defense performance.


## To Reproduce this result

- **Env setup and Model Preparation**:

  Run bash env.sh

- **Run SENTINEL generation**

  Run python pipeline_sentinel_gen.py and change the model/file in-place
  
- **Test ASR**
  Run python attack_success_rate.py and change the model/file in-place

- **Run SENTINEL over-refusal testing on or-bench (directly load from huggingface) /benign data (temp_data folder)**
  Run python orbench-sentinel.py and change the model/file in-place

- **Test FPR on or-bench**
  Run python over_refusal_rate.py and change the model/file in-place



# SENTINEL

This document presents the summarized performance of the SENTINEL defense against multiple jailbreak attacks on different models. The table below displays the Benign FPR and attack success rates (ASR) for various attack methods when using SENTINEL as a defense.

### SENTINEL Defense Results Across Models

| Model         | Benign FPR ↓ | OR-Bench FPR ↓ | GCG ↓ | PAIR ↓ | GPT-Fuzz ↓ | FewShot ↓ | AutoDAN ↓ |
|---------------|--------------|----------------|--------|---------|--------------|------------|-------------|
| Llama2-7b     | 3.05         | 34.02          | 5.66/3.32   | 2.33/3.01    | 1.00/2.45         | 4.33/5.51       | 0.00/1.12        |
| Llama3-8b     | 3.09         | 29.70          | 0.67/2.61   | 1.66/4.05    | 1.67/4.27         | 3.00/3.97       | 0.67/2.38        |
| Mistral-7b-v2 | 1.59         | 24.66          | 4.00/5.48   | 2.00/3.26    | 2.67/3.41         | 4.67/6.55       | 0.67/2.12        |
| Vicuna-7b     | 2.05         | 30.76          | 6.00/7.28   | 7.33/9.01    | 4.33/5.70         | 4.33/6.20       | 2.67/4.05        |

## Description
- **Benign FPR**: False positive rate on benign/boundary inputs.
- Lower values in Benign FPR indicate less over-refusal.
- **Jailbreak ASR**: The attack success rate by Harmbench Classifier. Lower values in the attack columns indicate stronger defense performance.


## To Reproduce this result

- **Env setup**:

  Run bash env.sh

- **For Main Results (ASR)**:
  
  Run bash defense_effectiveness.sh

 - **For Main Results (SR)**:
  
  python strong_reject.py 


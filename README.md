# SENTINEL

This document presents the summarized performance of the SENTINEL defense against multiple jailbreak attacks on different models. The table below displays the Benign FPR and attack success rates (ASR) for various attack methods when using SENTINEL as a defense.

## Summary Table

| Model   | Benign FPR | GCG Keywords | GCG Classifier | PAIR Keywords | PAIR Classifier | GPT-Fuzz Keywords | GPT-Fuzz Classifier | FewShot Keywords | FewShot Classifier | AutoDAN Keywords | AutoDAN Classifier |
|---------|------------|-------------|---------------|--------------|--------------|----------------|----------------|--------------|--------------|--------------|--------------|
| Vicuna  | 2.05       | 7.33        | 6.33          | 11.00        | 5.66         | 2.67           | 0.67           | 1.33         | 1.00         | 2.67         | 2.00         |
| Llama2  | 3.09       | 0.67        | 0.67          | 7.00         | 0.67         | 0.00           | 0.00           | 3.33         | 1.67         | 0.00         | 0.00         |
| Qwen    | 2.83       | 0.67        | 0.67          | 7.67         | 2.00         | 1.33           | 1.00           | 4.00         | 2.33         | 0.33         | 0.33         |
| Mistral | 4.79       | 18.00       | 15.33         | 27.33        | 18.00        | 6.33           | 2.00           | 22.33        | 17.33        | 1.67         | 1.67         |

## Description
- **Benign FPR**: False positive rate on benign inputs.
- Lower values in Benign FPR indicate less over-refusal.
- **Attack Columns**: Each attack method (GCG, PAIR, GPT-Fuzz, FewShot, AutoDAN) has two metrics—one for "Refusal Keywords" and one for Harmbench Classifier.
- Lower values in the attack columns indicate stronger defense performance.


## To Reproduce this result

- **Env setup and Model Preparation**:

  Run bash env.sh

- **Generate Harmfulness Scores and test FPR, ASR:**

  Run bash run.sh

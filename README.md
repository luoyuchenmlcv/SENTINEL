# SENTINEL

This document presents the summarized performance of the SENTINEL defense against multiple jailbreak attacks on different models. The table below displays the Benign FPR and attack success rates (ASR) for various attack methods when using SENTINEL as a defense.


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


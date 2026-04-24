# Logit

TF-IDF character n-gram features with logistic regression for DGA detection. Trained on 54 DGA families (~1M labeled domains).

**Architecture:** TF-IDF (char 3–5-grams) + Logistic Regression (SAGA) | **Params:** ~8M | **Framework:** scikit-learn  
**Trained model:** [reypapin/dga-logit](https://huggingface.co/reypapin/dga-logit)

## Contents

```
notebooks/
  dga_training_colab.ipynb               # Training and feature extraction
  Test_NEW_DGA.ipynb                     # Evaluation on unseen families

results/metrics/
  metricas_test_families_logit.csv       # Per-family metrics on 54 seen families
  metricas_new_families_logit.csv        # Per-family metrics on 25 unseen families
  final_test_metrics.json                # Aggregated metrics
```

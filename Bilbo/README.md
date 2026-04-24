# Bilbo

CNN + LSTM hybrid model for DGA detection (Highnam et al., 2021). Trained from scratch on 54 DGA families (~1M labeled domains).

**Architecture:** Conv1D + LSTM | **Params:** ~1.5M | **Framework:** PyTorch  
**Trained model:** [reypapin/dga-bilbo](https://huggingface.co/reypapin/dga-bilbo)

## Contents

```
notebooks/
  Bilbo_DGA_1M_Colab.ipynb              # Training on the full 1M dataset
  Test_NEW_DGA.ipynb                     # Evaluation on unseen families

results/metrics/
  metricas_test_families_bilbo.csv       # Per-family metrics on 54 seen families
  metricas_new_families_bilbo.csv        # Per-family metrics on 25 unseen families
  final_test_metrics_Bilbo.json          # Aggregated metrics
  classification_report.txt             # Sklearn classification report
```

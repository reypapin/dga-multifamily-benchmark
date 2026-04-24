# LA_Bin07

BiLSTM with sequential weighted attention for DGA detection (Leyva La O et al., 2024). Trained from scratch on 54 DGA families (~1M labeled domains).

**Architecture:** BiLSTM + SeqWeightedAttention | **Params:** ~8M | **Framework:** Keras  
**Trained model:** [reypapin/dga-labin07](https://huggingface.co/reypapin/dga-labin07)

## Contents

```
notebooks/
  Labin_1M.ipynb                         # Training on the full 1M dataset
  Test_NEW_DGA.ipynb                     # Evaluation on unseen families

scripts/
  compute_test_metrics.py                # Script to compute per-family metrics

results/metrics/
  metricas_test_families_labin.csv       # Per-family metrics on 54 seen families
  metricas_globales_final_labin.csv      # Per-family metrics on 25 unseen families (1:1)
  metricas_globales_final_labin_500.csv  # Unseen families at 1:10 ratio
  metricas_globales_final_labin_5000.csv # Unseen families at 1:100 ratio
```

# CNN

Character-level convolutional network for DGA detection. Trained from scratch on 54 DGA families (~1M labeled domains).

**Architecture:** Conv1D (64 filters) + MaxPool + FC | **Params:** ~50K | **Framework:** PyTorch  
**Trained model:** [reypapin/dga-cnn](https://huggingface.co/reypapin/dga-cnn)

## Contents

```
notebooks/
  CNN_Patron_1M.ipynb                    # Training on the full 1M dataset
  Test_NEW_DGA.ipynb                     # Evaluation on unseen families

scripts/
  compute_test_metrics.py                # Script to compute per-family metrics from raw predictions

results/metrics/
  metricas_test_families_cnn.csv         # Per-family metrics on 54 seen families
  metricas_globales_final_cnn.csv        # Per-family metrics on 25 unseen families
```

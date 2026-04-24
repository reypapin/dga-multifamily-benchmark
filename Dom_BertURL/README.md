# DomURLs-BERT

BERT-base encoder fine-tuned for DGA detection using LoRA (Mahdaouy et al., 2024). Fine-tuned on 54 DGA families (~1M labeled domains).

**Architecture:** BERT-base + LoRA (r=8, α=16) | **Params:** 110M | **Framework:** HuggingFace Transformers  
**Trained model:** [reypapin/dga-domurls-bert](https://huggingface.co/reypapin/dga-domurls-bert)

## Contents

```
notebooks/
  DomUrlBert.ipynb                              # Fine-tuning and evaluation

scripts/
  compute_test_metrics.py                       # Script to compute per-family metrics

results/metrics/
  metricas_test_families_domberturi.csv         # Per-family metrics on 54 seen families
  metricas_globales_final_domurlsbert.csv       # Per-family metrics on 25 unseen families (1:1)
  metricas_globales_final_domurlsbert_500.csv   # Unseen families at 1:10 ratio
  metricas_globales_final_domurlsbert_5000.csv  # Unseen families at 1:100 ratio
```

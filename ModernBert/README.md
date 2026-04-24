# ModernBERT

ModernBERT-base encoder fine-tuned for DGA detection. Fine-tuned on 54 DGA families (~1M labeled domains).

**Architecture:** ModernBERT-base | **Params:** 149M | **Framework:** HuggingFace Transformers  
**Trained model:** [reypapin/dga-modernbert](https://huggingface.co/reypapin/dga-modernbert)

## Contents

```
notebooks/
  ModernBERT_base_DGA_1M_Seba.ipynb            # Fine-tuning and evaluation

results/metrics/
  metricas_test_families_modernbert.csv         # Per-family metrics on 54 seen families
  metricas_globales_final_modernbert.csv        # Per-family metrics on 25 unseen families (1:1)
  metricas_globales_final_modernbert_500.csv    # Unseen families at 1:10 ratio
  metricas_globales_final_modernbert_5000.csv   # Unseen families at 1:100 ratio
```

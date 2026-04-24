# BiLSTM

Bidirectional LSTM with self-attention for DGA detection. Trained from scratch on 54 DGA families (~1M labeled domains).

**Architecture:** BiLSTM + self-attention | **Params:** ~700K | **Framework:** PyTorch  
**Trained model:** [reypapin/dga-bilstm](https://huggingface.co/reypapin/dga-bilstm)

## Contents

```
notebooks/
  BiLSTM_Attention_DGA_1M_Colab.ipynb   # Training on the full 1M dataset
  Test_NEW_DGA.ipynb                     # Evaluation on unseen families

results/metrics/
  metricas_test_families_bilstm.csv      # Per-family metrics on 54 seen families
  metricas_new_families_bilstm.csv       # Per-family metrics on 25 unseen families
  final_test_metrics_BiLSTM.json         # Aggregated metrics
  classification_report.txt             # Sklearn classification report
```

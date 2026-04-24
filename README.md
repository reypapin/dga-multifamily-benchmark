# A Training and Evaluation Methodology for Robust DGA Detection

**Reynier Leyva La O · Carlos A. Catania**  
GridTICs / Universidad Tecnológica Nacional · LABSIN / Universidad Nacional de Cuyo · CONICET  
ARGENCON 2026 (IEEE)

---

## Overview

This repository contains the paper source, evaluation code, and results for a benchmark study of seven DGA detection models across 79 malware families.

The study introduces an evaluation methodology organized around four scenarios:

| Scenario | Families | Type |
|---|---|---|
| S1 | 54 seen | All |
| S2 | 54 seen | Word list only |
| S3 | 25 unseen | All |
| S4 | 25 unseen | Word list only (hardest) |

Inference time and performance under class imbalance (1:1, 1:10, 1:100) are reported as deployment considerations.

---

## Models

All seven trained models are available on Hugging Face:

| Model | Architecture | HuggingFace |
|---|---|---|
| CNN | Conv1D + MaxPool | [reypapin/dga-cnn](https://huggingface.co/reypapin/dga-cnn) |
| BiLSTM | BiLSTM + self-attention | [reypapin/dga-bilstm](https://huggingface.co/reypapin/dga-bilstm) |
| Bilbo | CNN + LSTM | [reypapin/dga-bilbo](https://huggingface.co/reypapin/dga-bilbo) |
| Logit | TF-IDF + Logistic Regression | [reypapin/dga-logit](https://huggingface.co/reypapin/dga-logit) |
| LA\_Bin07 | BiLSTM + SeqWeightedAttention | [reypapin/dga-labin07](https://huggingface.co/reypapin/dga-labin07) |
| DomURLs-BERT | BERT-base + LoRA | [reypapin/dga-domurls-bert](https://huggingface.co/reypapin/dga-domurls-bert) |
| ModernBERT | ModernBERT-base fine-tuned | [reypapin/dga-modernbert](https://huggingface.co/reypapin/dga-modernbert) |

---

## Repository Structure

```
├── argencon_paper.tex       # Paper source (LaTeX)
├── cas-refs.bib             # Bibliography
├── statistical_tests.py     # Friedman + Wilcoxon significance tests
├── compute_imbalance_metrics.py  # Class imbalance analysis
├── statistical_results/     # Precomputed statistical test outputs
├── BiLSTM/results/metrics/  # Per-family F1 metrics
├── Bilbo/results/metrics/
├── CNN/results/metrics/
├── Dom_BertURL/results/metrics/
├── Labin/results/metrics/
├── Logit/results/metrics/
└── ModernBert/results/metrics/
```

---

## Key Results

| Model | F1 (Seen) | Rank | F1 (Unseen) | Rank |
|---|---|---|---|---|
| ModernBERT | 0.958 | 1 | 0.691 | 6 |
| LA\_Bin07 | 0.943 | 2 | 0.652 | 7 |
| DomURLs-BERT | 0.905 | 3 | 0.822 | 1 |
| Logit | 0.903 | 4 | 0.804 | 2 |
| CNN | 0.901 | 5 | 0.745 | 3 |
| Bilbo | 0.900 | 6 | 0.721 | 5 |
| BiLSTM | 0.856 | 7 | 0.736 | 4 |

Under a 1:100 DGA-to-legitimate traffic ratio on unseen word list families, the best result across all models is **F1 = 0.182**.

---

## Citation

```bibtex
@inproceedings{leyva2026dga,
  author    = {Leyva La O, Reynier and Catania, Carlos A.},
  title     = {A Training and Evaluation Methodology for Robust {DGA} Detection},
  booktitle = {ARGENCON 2026},
  year      = {2026},
  publisher = {IEEE}
}
```

# Statistical Results

Precomputed outputs from `statistical_tests.py`. All tests follow Demšar (2006) using mean F1 per family as the unit of observation.

## Files

| File | Description |
|---|---|
| `mean_f1_seen.csv` | Mean F1 per family for each model on the 54 seen families |
| `mean_f1_unseen.csv` | Mean F1 per family for each model on the 25 unseen families |
| `wilcoxon_seen.csv` | Pairwise Wilcoxon signed-rank tests with Bonferroni correction (seen families) |
| `wilcoxon_unseen.csv` | Pairwise Wilcoxon signed-rank tests with Bonferroni correction (unseen families) |
| `imbalance_extended.csv` | F1, Precision, and FPR under 1:1, 1:10, and 1:100 class ratios on unseen word list families |

## Reproducing

```bash
python statistical_tests.py
```

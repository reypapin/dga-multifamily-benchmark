"""
Statistical significance tests for DGA detector comparison.

Unit of observation: mean F1 per family.
  - Seen (54 families): from precomputed metrics CSVs (all 7 models)
  - Unseen (25 families): from precomputed CSVs (BiLSTM/Bilbo/Logit)
                          and raw prediction files (CNN/DomURLs-BERT/LA_Bin07/ModernBERT)

Tests:
  1. Friedman test (global, 7 models)
  2. Pairwise Wilcoxon signed-rank (two-sided) + Bonferroni correction

Reference: Demsar (2006), JMLR 7:1-30.
"""

import os, re, gzip
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations

BASE = "/home/reynier/Work/Doc_Leer/New_Paper_ModernBert_HF"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Models with precomputed CSVs for both seen and unseen
SEEN_CSVS = {
    "BiLSTM":      ("BiLSTM/results/metrics/metricas_test_families_bilstm.csv",   "F1 Score Mean"),
    "Bilbo":       ("Bilbo/results/metrics/metricas_test_families_bilbo.csv",      "F1 Score Mean"),
    "CNN":         ("CNN/results/metrics/metricas_test_families_cnn.csv",          "F1-Score"),
    "DomURLs-BERT":("Dom_BertURL/results/metrics/metricas_test_families_domberturi.csv", "F1-Score"),
    "LA_Bin07":    ("Labin/results/metrics/metricas_test_families_labin.csv",      "F1-Score"),
    "Logit":       ("Logit/results/metrics/metricas_test_families_logit.csv",      "F1 Score Mean"),
    "ModernBERT":  ("ModernBert/results/metrics/metricas_test_families_modernbert.csv", "F1 Score Mean"),
}

UNSEEN_CSVS = {
    "BiLSTM": ("BiLSTM/results/metrics/metricas_new_families_bilstm.csv", "F1-Score"),
    "Bilbo":  ("Bilbo/results/metrics/metricas_new_families_bilbo.csv",   "F1-Score"),
    "Logit":  ("Logit/results/metrics/metricas_new_families_logit.csv",   "F1-Score"),
}

# Models with raw prediction files for unseen families
UNSEEN_RAW = {
    "CNN": {
        "dir": "CNN/results/raw/new_families",
        "pat": r"results_cnn_(?P<family>[^.]+)\.gz_(?P<run>\d+)\.csv\.gz",
    },
    "DomURLs-BERT": {
        "dir": "Dom_BertURL/results/raw/new_families",
        "pat": r"results_domurlsbert_(?P<family>[^.]+)\.gz_(?P<run>\d+)\.csv\.gz",
    },
    "LA_Bin07": {
        "dir": "Labin/results/raw/new_families",
        "pat": r"results_labin_(?P<family>[^.]+)\.gz_(?P<run>\d+)\.csv\.gz",
    },
    "ModernBERT": {
        "dir": "ModernBert/results/raw/new_families",
        "pat": r"results_modernbert_(?P<family>[^.]+)\.gz_(?P<run>\d+)\.csv\.gz",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv_f1(csv_rel, f1_col):
    """Load mean F1 per family from a precomputed metrics CSV."""
    df = pd.read_csv(os.path.join(BASE, csv_rel))
    fam_col = [c for c in df.columns if c.lower() == "family"][0]
    df[fam_col] = df[fam_col].str.lower().str.strip()
    # Drop aggregate summary rows (not actual families)
    df = df[~df[fam_col].isin(["global_mean", "mean", "average", "total"])]
    return df.set_index(fam_col)[f1_col]


def to_binary(series):
    if pd.api.types.is_string_dtype(series):
        return series.str.lower().isin(["dga", "1", "true"]).astype(int)
    return series.astype(int)


def f1_from_gz(filepath):
    with gzip.open(filepath, "rt") as f:
        df = pd.read_csv(f)
    y_true = to_binary(df["label"])
    y_pred = to_binary(df["pred"])
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def compute_mean_f1_from_raw(directory, pattern):
    """Compute mean F1 per family over all runs from raw gz files."""
    regex = re.compile(pattern)
    family_runs = {}
    for fname in os.listdir(directory):
        m = regex.fullmatch(fname)
        if m:
            fam = m.group("family")
            run = int(m.group("run"))
            family_runs.setdefault(fam, []).append((run, os.path.join(directory, fname)))

    result = {}
    for fam, run_files in sorted(family_runs.items()):
        f1s = []
        for _, fpath in sorted(run_files):
            try:
                f1s.append(f1_from_gz(fpath))
            except Exception as e:
                f1s.append(np.nan)
        result[fam] = np.nanmean(f1s)
    return pd.Series(result)


# ---------------------------------------------------------------------------
# Build F1 matrices
# ---------------------------------------------------------------------------

def build_seen_matrix():
    print("Loading SEEN families (precomputed CSVs)...")
    series = {}
    for model, (csv, col) in SEEN_CSVS.items():
        try:
            series[model] = load_csv_f1(csv, col)
            print(f"  {model}: {len(series[model])} families")
        except Exception as e:
            print(f"  WARNING {model}: {e}")
    return pd.DataFrame(series)


def build_unseen_matrix():
    print("Loading UNSEEN families...")
    series = {}
    for model, (csv, col) in UNSEEN_CSVS.items():
        try:
            series[model] = load_csv_f1(csv, col)
            print(f"  {model} (CSV): {len(series[model])} families")
        except Exception as e:
            print(f"  WARNING {model}: {e}")
    for model, cfg in UNSEEN_RAW.items():
        try:
            s = compute_mean_f1_from_raw(
                os.path.join(BASE, cfg["dir"]), cfg["pat"]
            )
            series[model] = s
            print(f"  {model} (raw): {len(s)} families")
        except Exception as e:
            print(f"  WARNING {model}: {e}")
    return pd.DataFrame(series)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def friedman_test(df):
    stat, p = stats.friedmanchisquare(*[df[col].values for col in df.columns])
    return stat, p


def pairwise_wilcoxon(df):
    models = list(df.columns)
    n_pairs = len(list(combinations(models, 2)))
    rows = []
    for a, b in combinations(models, 2):
        paired = df[[a, b]].dropna()
        x, y = paired[a].values, paired[b].values
        n = len(x)
        if n < 5:
            rows.append(dict(model_a=a, model_b=b, n=n,
                             mean_a=np.nan, mean_b=np.nan,
                             stat=np.nan, p_raw=np.nan, p_bonf=np.nan, sig=False))
            continue
        stat, p = stats.wilcoxon(x, y, alternative="two-sided")
        rows.append(dict(model_a=a, model_b=b, n=n,
                         mean_a=x.mean(), mean_b=y.mean(),
                         stat=stat, p_raw=p))
    result = pd.DataFrame(rows)
    result["p_bonf"] = (result["p_raw"] * n_pairs).clip(upper=1.0)
    result["sig"] = result["p_bonf"] < 0.05
    return result.sort_values("p_bonf").reset_index(drop=True)


def run_analysis(label, df):
    print(f"\n{'='*65}")
    print(f"AXIS: {label}")
    print(f"{'='*65}")

    df_full = df.dropna()
    n_fam = len(df_full)
    n_mod = len(df_full.columns)
    print(f"Models: {n_mod}  |  Families with complete data: {n_fam}")

    print(f"\nMean F1 per model (ranked):")
    means = df_full.mean().sort_values(ascending=False)
    for i, (m, v) in enumerate(means.items(), 1):
        print(f"  {i}. {m:20s}: {v:.4f}")

    stat, p = friedman_test(df_full)
    print(f"\nFriedman test: chi2({n_mod-1}) = {stat:.4f},  p = {p:.2e}")
    if p < 0.05:
        print("  SIGNIFICANT — post-hoc Wilcoxon tests are justified.")
    else:
        print("  Not significant (p >= 0.05).")

    pw = pairwise_wilcoxon(df_full)
    n_sig = pw["sig"].sum()
    print(f"\nPairwise Wilcoxon + Bonferroni ({len(pw)} pairs, alpha=0.05):")
    print(f"  Significant pairs: {n_sig}/{len(pw)}\n")
    print(pw[["model_a","model_b","n","mean_a","mean_b","p_raw","p_bonf","sig"]]
          .to_string(index=False, float_format="%.4f"))

    return df_full, pw


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    seen_df  = build_seen_matrix()
    unseen_df = build_unseen_matrix()

    mean_seen,   pw_seen   = run_analysis("SEEN FAMILIES (54 training families)",       seen_df)
    mean_unseen, pw_unseen = run_analysis("UNSEEN FAMILIES (25 generalization families)", unseen_df)

    out = os.path.join(BASE, "statistical_results")
    os.makedirs(out, exist_ok=True)
    mean_seen.to_csv(f"{out}/mean_f1_seen.csv")
    mean_unseen.to_csv(f"{out}/mean_f1_unseen.csv")
    pw_seen.to_csv(f"{out}/wilcoxon_seen.csv", index=False)
    pw_unseen.to_csv(f"{out}/wilcoxon_unseen.csv", index=False)
    print(f"\nResults saved to {out}/")

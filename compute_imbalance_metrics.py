"""
Compute F1, Precision, and FPR for the 4 unseen word list families
at three DGA-to-legitimate ratios: 1:1 (50+50), 1:10 (50+450), 1:100 (50+4950).

Sources:
  - BiLSTM/Bilbo/Logit/CNN: raw gz files in test_05k_5k/
  - DomURLs-BERT/LA_Bin07/ModernBERT: precomputed CSV (500, 5000) + WL raw for 1:1
  - 1:1 ratio for all models: from unseen new_families raw (bigviktor/gozi_rfc4343/pizd/ngioweb)
"""

import os, re, gzip
import numpy as np
import pandas as pd

BASE = "/home/reynier/Work/Doc_Leer/New_Paper_ModernBert_HF"
WL_FAMILIES = ["bigviktor", "gozi_rfc4343", "ngioweb", "pizd"]

# ---------------------------------------------------------------------------
# Raw file configurations per model per ratio
# ratio_key: "500" = 1:10,  "5000" = 1:100,  "50" = 1:1
# ---------------------------------------------------------------------------
RAW_CONFIGS = {
    "BiLSTM": {
        "dir_500":  "BiLSTM/results/raw/test_05k_5k",
        "pat_500":  r"results_bilstm_500_(?P<family>[^.]+)\.gz_(?P<run>\d+)\.csv\.gz",
        "dir_5000": "BiLSTM/results/raw/test_05k_5k",
        "pat_5000": r"results_bilstm_5000_(?P<family>[^.]+)\.gz_(?P<run>\d+)\.csv\.gz",
        "dir_50":   "BiLSTM/results/raw/new_families",
        "pat_50":   r"results_bilstm_(?P<family>[^.]+)\.gz_(?P<run>\d+)\.csv\.gz",
    },
    "Bilbo": {
        "dir_500":  "Bilbo/results/raw/test_05k_5k",
        "pat_500":  r"results_bilbo_500_(?P<family>[^.]+)\.gz_(?P<run>\d+)\.csv\.gz",
        "dir_5000": "Bilbo/results/raw/test_05k_5k",
        "pat_5000": r"results_bilbo_5000_(?P<family>[^.]+)\.gz_(?P<run>\d+)\.csv\.gz",
        "dir_50":   "Bilbo/results/raw/new_families",
        "pat_50":   r"results_bilbo_(?P<family>[^.]+)\.gz_(?P<run>\d+)\.csv\.gz",
    },
    "CNN": {
        "dir_500":  "CNN/results/raw/test_05k_5k",
        "pat_500":  r"results_cnn_500_(?P<family>[^.]+)\.gz_(?P<run>\d+)\.csv\.gz",
        "dir_5000": "CNN/results/raw/test_05k_5k",
        "pat_5000": r"results_cnn_5000_(?P<family>[^.]+)\.gz_(?P<run>\d+)\.csv\.gz",
        "dir_50":   "CNN/results/raw/new_families",
        "pat_50":   r"results_cnn_(?P<family>[^.]+)\.gz_(?P<run>\d+)\.csv\.gz",
    },
    "DomURLs-BERT": {
        "csv_500":  "Dom_BertURL/results/metrics/metricas_globales_final_domurlsbert_500.csv",
        "csv_5000": "Dom_BertURL/results/metrics/metricas_globales_final_domurlsbert_5000.csv",
        "dir_50":   "Dom_BertURL/results/raw/new_families",
        "pat_50":   r"results_domurlsbert_(?P<family>[^.]+)\.gz_(?P<run>\d+)\.csv\.gz",
    },
    "LA_Bin07": {
        "csv_500":  "Labin/results/metrics/metricas_globales_final_labin_500.csv",
        "csv_5000": "Labin/results/metrics/metricas_globales_final_labin_5000.csv",
        "dir_50":   "Labin/results/raw/new_families",
        "pat_50":   r"results_labin_(?P<family>[^.]+)\.gz_(?P<run>\d+)\.csv\.gz",
    },
    "Logit": {
        "dir_500":  "Logit/results/raw/test_05k_5k",
        "pat_500":  r"results_logit_500_(?P<family>[^.]+)\.gz_(?P<run>\d+)\.csv\.gz",
        "dir_5000": "Logit/results/raw/test_05k_5k",
        "pat_5000": r"results_logit_5000_(?P<family>[^.]+)\.gz_(?P<run>\d+)\.csv\.gz",
        "dir_50":   "Logit/results/raw/new_families",
        "pat_50":   r"results_logit_(?P<family>[^.]+)\.gz_(?P<run>\d+)\.csv\.gz",
    },
    "ModernBERT": {
        "csv_500":  "ModernBert/results/metrics/metricas_globales_final_modernbert_500.csv",
        "csv_5000": "ModernBert/results/metrics/metricas_globales_final_modernbert_5000.csv",
        "dir_50":   "ModernBert/results/raw/new_families",
        "pat_50":   r"results_modernbert_(?P<family>[^.]+)\.gz_(?P<run>\d+)\.csv\.gz",
    },
}


def to_binary(series):
    if pd.api.types.is_string_dtype(series):
        return series.str.lower().isin(["dga", "1", "true"]).astype(int)
    return series.astype(int)


def metrics_from_gz(filepath):
    """Return (f1, precision, fpr) from a single run file."""
    with gzip.open(filepath, "rt") as f:
        df = pd.read_csv(f)
    y_true = to_binary(df["label"])
    y_pred = to_binary(df["pred"])
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return f1, prec, fpr


def compute_from_raw(directory, pattern, families=None):
    """
    Compute mean F1, Precision, FPR per family from raw gz files.
    Returns DataFrame: rows=families, cols=[F1, Precision, FPR].
    """
    regex = re.compile(pattern)
    family_runs = {}
    for fname in os.listdir(directory):
        m = regex.fullmatch(fname)
        if m:
            fam = m.group("family")
            if families and fam not in families:
                continue
            run = int(m.group("run"))
            family_runs.setdefault(fam, []).append((run, os.path.join(directory, fname)))

    records = {}
    for fam, run_files in sorted(family_runs.items()):
        f1s, precs, fprs = [], [], []
        for _, fpath in sorted(run_files):
            try:
                f1, prec, fpr = metrics_from_gz(fpath)
                f1s.append(f1); precs.append(prec); fprs.append(fpr)
            except Exception:
                pass
        if f1s:
            records[fam] = {
                "F1": np.mean(f1s),
                "Precision": np.mean(precs),
                "FPR": np.mean(fprs),
            }
    return pd.DataFrame(records).T


def load_from_csv(csv_rel, families=None):
    """Load F1, Precision, FPR from a precomputed metrics CSV."""
    df = pd.read_csv(os.path.join(BASE, csv_rel))
    fam_col = [c for c in df.columns if c.lower() == "family"][0]
    df[fam_col] = df[fam_col].str.lower().str.strip()
    df = df[~df[fam_col].isin(["global_mean", "mean", "average", "total"])]
    if families:
        df = df[df[fam_col].isin(families)]
    df = df.set_index(fam_col)

    # Normalize column names
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if "f1" in cl and "std" not in cl:
            col_map[c] = "F1"
        elif "precision" in cl and "std" not in cl:
            col_map[c] = "Precision"
        elif "fpr" in cl and "std" not in cl:
            col_map[c] = "FPR"
    return df.rename(columns=col_map)[["F1", "Precision", "FPR"]]


def get_model_ratio(model, ratio_key):
    """Get metrics DataFrame for a model at a given ratio ('50', '500', '5000')."""
    cfg = RAW_CONFIGS[model]
    if f"csv_{ratio_key}" in cfg:
        return load_from_csv(cfg[f"csv_{ratio_key}"], families=WL_FAMILIES)
    elif f"dir_{ratio_key}" in cfg:
        return compute_from_raw(
            os.path.join(BASE, cfg[f"dir_{ratio_key}"]),
            cfg[f"pat_{ratio_key}"],
            families=WL_FAMILIES
        )
    return pd.DataFrame()


def mean_over_wl(df):
    """Average F1, Precision, FPR over the 4 WL families."""
    df = df[df.index.isin(WL_FAMILIES)]
    return df[["F1", "Precision", "FPR"]].mean()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    MODEL_ORDER = ["BiLSTM", "Bilbo", "CNN", "LA_Bin07", "DomURLs-BERT", "Logit", "ModernBERT"]
    RATIOS = {"50": "1:1", "500": "1:10", "5000": "1:100"}

    results = {}  # model -> ratio_label -> {F1, Precision, FPR}

    for model in MODEL_ORDER:
        results[model] = {}
        for ratio_key, ratio_label in RATIOS.items():
            df = get_model_ratio(model, ratio_key)
            if df.empty:
                print(f"  WARNING: no data for {model} at {ratio_label}")
                results[model][ratio_label] = {"F1": np.nan, "Precision": np.nan, "FPR": np.nan}
            else:
                means = mean_over_wl(df)
                results[model][ratio_label] = means.to_dict()
                print(f"  {model} {ratio_label}: F1={means['F1']:.3f}  Prec={means['Precision']:.3f}  FPR={means['FPR']:.4f}")

    print("\n\n=== TABLE VI EXTENDED (LaTeX) ===\n")
    print(r"\begin{table}[!t]")
    print(r"\caption{F1, Precision, and FPR on the four unseen word list families under")
    print(r"three DGA-to-legitimate ratios (1:1, 1:10, 1:100). Each value is averaged")
    print(r"over 30 runs per family and then over the four families.")
    print(r"Models sorted by F1 at 1:1. Best value per metric and ratio in \textbf{bold}.}")
    print(r"\label{tab:imbalance}")
    print(r"\centering")
    print(r"\renewcommand{\arraystretch}{1.1}")
    print(r"\resizebox{\linewidth}{!}{%")
    print(r"\begin{tabular}{l rrr rrr rrr}")
    print(r"\toprule")
    print(r"& \multicolumn{3}{c}{\textbf{1:1}} & \multicolumn{3}{c}{\textbf{1:10}} & \multicolumn{3}{c}{\textbf{1:100}} \\")
    print(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}")
    print(r"\textbf{Model} & F1 & Prec & FPR & F1 & Prec & FPR & F1 & Prec & FPR \\")
    print(r"\midrule")

    # Find best per column for bolding
    for metric_col, col_label in [("F1","F1"), ("Precision","Prec"), ("FPR","FPR")]:
        pass  # compute after collecting all data

    # Collect data matrix
    rows_data = []
    for model in MODEL_ORDER:
        row = {"model": model}
        for ratio_label in ["1:1", "1:10", "1:100"]:
            for metric in ["F1", "Precision", "FPR"]:
                row[f"{ratio_label}_{metric}"] = results[model][ratio_label].get(metric, np.nan)
        rows_data.append(row)

    df_out = pd.DataFrame(rows_data).set_index("model")

    # Best per column (F1/Prec: max, FPR: min)
    best = {}
    for ratio_label in ["1:1", "1:10", "1:100"]:
        for metric in ["F1", "Precision"]:
            col = f"{ratio_label}_{metric}"
            best[col] = df_out[col].idxmax()
        col = f"{ratio_label}_FPR"
        best[col] = df_out[col].idxmin()

    for model in MODEL_ORDER:
        row = df_out.loc[model]
        cells = []
        for ratio_label in ["1:1", "1:10", "1:100"]:
            for metric in ["F1", "Precision", "FPR"]:
                col = f"{ratio_label}_{metric}"
                val = row[col]
                fmt = f"{val:.3f}" if metric != "FPR" else f"{val:.4f}"
                if best.get(col) == model:
                    fmt = r"\textbf{" + fmt + "}"
                cells.append(fmt)
        model_tex = model.replace("_", r"\_")
        if model == "ModernBERT":
            model_tex = r"\textbf{ModernBERT}"
        print(f"{model_tex} & " + " & ".join(cells) + r" \\")

    print(r"\bottomrule")
    print(r"\end{tabular}}")
    print(r"\end{table}")

    # Also save CSV
    out = os.path.join(BASE, "statistical_results")
    os.makedirs(out, exist_ok=True)
    df_out.round(4).to_csv(f"{out}/imbalance_extended.csv")
    print(f"\nSaved to {out}/imbalance_extended.csv")

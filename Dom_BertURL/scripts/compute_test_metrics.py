import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
MODEL_NAME = 'LORA_classifier'
RUNS = 30

def fpr_tpr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn + 1e-10)
    tpr = tp / (tp + fn + 1e-10)
    return fpr, tpr

all_files = os.listdir(RESULTS_DIR)
families = sorted(set(
    '_'.join(f.replace(f'results_{MODEL_NAME}_', '').rsplit('_', 1)[:-1])
    for f in all_files
    if f.startswith(f'results_{MODEL_NAME}_') and f.endswith('.csv.gz')
))

print(f"Familias encontradas: {len(families)}")

all_acc, all_acc_std = [], []
all_pre, all_pre_std = [], []
all_rec, all_rec_std = [], []
all_f1,  all_f1_std  = [], []
all_fpr, all_fpr_std = [], []
all_tpr, all_tpr_std = [], []
all_qt,  all_qts     = [], []
family_names = []

for family in families:
    acc, pre, rec, f1, fpr, tpr, qt = [], [], [], [], [], [], []

    for run in range(RUNS):
        path = os.path.join(RESULTS_DIR, f'results_{MODEL_NAME}_{family}_{run}.csv.gz')
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)
        # label is already 0/1 (1 = dga)
        y_true = df['label'].astype(int)
        y_pred = df['pred'].astype(int)

        acc.append(accuracy_score(y_true, y_pred))
        pre.append(precision_score(y_true, y_pred, zero_division=0))
        rec.append(recall_score(y_true, y_pred, zero_division=0))
        f1.append(f1_score(y_true, y_pred, zero_division=0))

        fpr_val, tpr_val = fpr_tpr(y_true, y_pred)
        fpr.append(fpr_val)
        tpr.append(tpr_val)
        if 'query_time' in df.columns:
            qt.append(df['query_time'].mean())

    if acc:
        family_names.append(family.replace('.gz', ''))
        all_acc.append(np.mean(acc));   all_acc_std.append(np.std(acc))
        all_pre.append(np.mean(pre));   all_pre_std.append(np.std(pre))
        all_rec.append(np.mean(rec));   all_rec_std.append(np.std(rec))
        all_f1.append(np.mean(f1));     all_f1_std.append(np.std(f1))
        all_fpr.append(np.mean(fpr));   all_fpr_std.append(np.std(fpr))
        all_tpr.append(np.mean(tpr));   all_tpr_std.append(np.std(tpr))
        all_qt.append(np.mean(qt) if qt else 0.0)
        all_qts.append(np.std(qt) if qt else 0.0)

        print(f'{family_names[-1]:20}: '
              f'acc:{all_acc[-1]:.2f}±{all_acc_std[-1]:.3f} '
              f'f1:{all_f1[-1]:.2f}±{all_f1_std[-1]:.3f} '
              f'pre:{all_pre[-1]:.2f}±{all_pre_std[-1]:.3f} '
              f'rec:{all_rec[-1]:.2f}±{all_rec_std[-1]:.3f} '
              f'FPR:{all_fpr[-1]:.2f}±{all_fpr_std[-1]:.3f} '
              f'TPR:{all_tpr[-1]:.2f}±{all_tpr_std[-1]:.3f}')

df_metrics = pd.DataFrame({
    'Family':          family_names,
    'Accuracy':        all_acc,
    'Acc_std':         all_acc_std,
    'Precision':       all_pre,
    'Pre_std':         all_pre_std,
    'Recall':          all_rec,
    'Rec_std':         all_rec_std,
    'F1-Score':        all_f1,
    'F1_std':          all_f1_std,
    'FPR':             all_fpr,
    'FPR_std':         all_fpr_std,
    'TPR':             all_tpr,
    'TPR_std':         all_tpr_std,
    'Query_Time_Mean': all_qt,
    'Query_Time_Std':  all_qts,
})

global_means = df_metrics.mean(numeric_only=True).to_dict()
global_means['Family'] = 'GLOBAL_MEAN'
df_metrics = pd.concat([df_metrics, pd.DataFrame([global_means])], ignore_index=True)

output_path = os.path.join(os.path.dirname(__file__), 'metricas_test_families_domberturi.csv')
df_metrics.to_csv(output_path, index=False)
print(f"\n✅ CSV guardado en: {output_path}")

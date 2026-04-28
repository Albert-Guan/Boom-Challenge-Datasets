import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, f1_score

from features import build_features, transform_targets, invert_targets
from models import make_model, ALL_TARGETS


def train_and_validate(data_dir='forward_prediction', n_splits=5, seed=42):
    X_raw = pd.read_csv(f'{data_dir}/train.csv').reset_index(drop=True)
    y_raw = pd.read_csv(f'{data_dir}/train_labels.csv').reset_index(drop=True)
    if len(X_raw) != len(y_raw):
        raise ValueError(
            f'{data_dir}/train.csv ({len(X_raw)} rows) and '
            f'{data_dir}/train_labels.csv ({len(y_raw)} rows) must align'
        )

    X = build_features(X_raw)
    y = transform_targets(y_raw, X['L_char'])

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    physical_cols = ['P80', 'R95', 'R50_fines', 'R50_oversize', 'fines_frac', 'oversize_frac']
    physical_scores = {c: [] for c in physical_cols}
    # F1 on above/below training-median splits (regression targets binarized per fold)
    physical_f1 = {c: [] for c in physical_cols}

    for fold, (tr, va) in enumerate(kf.split(X)):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr = y.iloc[tr]
        y_tr_phys = y_raw.iloc[tr].reset_index(drop=True)

        preds_transformed = pd.DataFrame(index=X_va.index)
        for target in ALL_TARGETS:
            model = make_model(target).fit(X_tr, y_tr[target])
            preds_transformed[target] = model.predict(X_va)

        preds_physical = invert_targets(preds_transformed, X_va['L_char']).reset_index(drop=True)
        y_va_phys = y_raw.iloc[va].reset_index(drop=True)
        for col in physical_cols:
            physical_scores[col].append(
                np.sqrt(mean_squared_error(y_va_phys[col], preds_physical[col]))
            )
            thr = np.median(y_tr_phys[col])
            y_bin = (y_va_phys[col].to_numpy() > thr).astype(int)
            p_bin = (preds_physical[col].to_numpy() > thr).astype(int)
            physical_f1[col].append(f1_score(y_bin, p_bin, zero_division=0))
        print(
            f"Fold {fold} RMSE: " + ", ".join(f"{c}={physical_scores[c][-1]:.4f}" for c in physical_cols)
        )
        print(
            f"Fold {fold} F1:   " + ", ".join(f"{c}={physical_f1[c][-1]:.4f}" for c in physical_cols)
        )

    print("\nMean CV RMSE (physical units):")
    for c in physical_cols:
        print(f"  {c}: {np.mean(physical_scores[c]):.4f} ± {np.std(physical_scores[c]):.4f}")

    print("\nMean CV F1 (median split vs training fold, physical units):")
    for c in physical_cols:
        print(f"  {c}: {np.mean(physical_f1[c]):.4f} ± {np.std(physical_f1[c]):.4f}")

    rmse_array = np.column_stack([physical_scores[c] for c in physical_cols])
    f1_array = np.column_stack([physical_f1[c] for c in physical_cols])

    # Final models on full data
    final_models = {t: make_model(t).fit(X, y[t]) for t in ALL_TARGETS}
    for t in ALL_TARGETS:
        print(f"\n{t} baseline coefficients:\n{final_models[t].baseline_coefficients()}")

    joblib.dump(final_models, 'models.joblib')

    fold_idx = np.arange(n_splits)
    fig, (ax_rmse, ax_f1) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for j, col in enumerate(physical_cols):
        ax_rmse.plot(fold_idx, rmse_array[:, j], marker='o', label=col)
        ax_f1.plot(fold_idx, f1_array[:, j], marker='o', label=col)
    ax_rmse.set_ylabel('RMSE')
    ax_rmse.set_title('Cross-validation RMSE by fold')
    ax_rmse.legend(loc='best', ncol=2, fontsize=8)
    ax_rmse.grid(True, alpha=0.3)
    ax_f1.set_xlabel('Fold')
    ax_f1.set_ylabel('F1')
    ax_f1.set_title('Cross-validation F1 by fold (median split)')
    ax_f1.legend(loc='best', ncol=2, fontsize=8)
    ax_f1.grid(True, alpha=0.3)
    ax_f1.set_xticks(fold_idx)
    plt.tight_layout()
    plt.savefig('cv_metrics.png', dpi=150)
    print("\nSaved CV metric plot to cv_metrics.png")
    plt.show()

    return final_models


if __name__ == '__main__':
    train_and_validate()

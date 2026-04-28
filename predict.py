import pandas as pd
import joblib
from features import build_features, invert_targets
from models import ALL_TARGETS


def generate_submission(data_dir='forward_prediction',
                        model_path='models.joblib',
                        output_path='submission/prediction_submission.csv'):
    X_test_raw = pd.read_csv(f'{data_dir}/test.csv')
    X_test = build_features(X_test_raw)
    models = joblib.load(model_path)

    preds_transformed = pd.DataFrame(index=X_test.index)
    for t in ALL_TARGETS:
        preds_transformed[t] = models[t].predict(X_test)

    preds = invert_targets(preds_transformed, X_test['L_char'])

    # Constraint: fines_frac + oversize_frac ≤ 1
    total = preds['fines_frac'] + preds['oversize_frac']
    mask = total > 0.999
    if mask.any():
        scale = 0.999 / total[mask]
        preds.loc[mask, 'fines_frac']    *= scale
        preds.loc[mask, 'oversize_frac'] *= scale
        print(f"Rescaled fractions for {mask.sum()} rows")

    # Constraint: R50_fines ≤ R95
    mask = preds['R50_fines'] > preds['R95']
    if mask.any():
        preds.loc[mask, 'R50_fines'] = preds.loc[mask, 'R95']
        print(f"Clipped R50_fines for {mask.sum()} rows")

    submission = pd.DataFrame({
        'scenario_id': range(len(X_test)),
        'P80':           preds['P80'],
        'fines_frac':    preds['fines_frac'],
        'oversize_frac': preds['oversize_frac'],
        'R95':           preds['R95'],
        'R50_fines':     preds['R50_fines'],
        'R50_oversize':  preds['R50_oversize'],
    })
    submission.to_csv(output_path, index=False)
    print(f"Wrote {len(submission)} rows to {output_path}")


if __name__ == '__main__':
    generate_submission()

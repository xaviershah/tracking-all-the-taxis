import os
import numpy as np
import pandas as pd

DATA_PATH = '/home/chongshengwang/ads/all_types_with_fhvhv_dense_filtered_holidays.parquet'
PRED_DIR  = '/home/chongshengwang/ads/short_term_prediction/outputs/model_comparison_no_fhv_2024/dm_predictions'
OUT_DIR   = '/home/chongshengwang/ads/short_term_prediction/outputs/model_comparison_no_fhv_2024'

TEST_MONTHS = [4, 5, 6, 7, 8, 9, 10, 11, 12]

df = pd.read_parquet(DATA_PATH, columns=['trip_count', 'year', 'month', 'cab_type'])
df = df[df['cab_type'] != 'fhv'].reset_index(drop=True)

year_arr  = df['year'].to_numpy()
month_arr = df['month'].to_numpy()
test_mask = (year_arr == 2024) & np.isin(month_arr, TEST_MONTHS)

test_df   = df.loc[test_mask].reset_index(drop=True)
y_test    = test_df['trip_count'].values.astype(np.float64)
cab_types = test_df['cab_type'].values
del df

cab_masks = {
    'fhvhv':  cab_types == 'fhvhv',
    'yellow': cab_types == 'yellow',
    'green':  cab_types == 'green',
    'all':    np.ones(len(y_test), dtype=bool),
}

print(f"  Total test rows: {len(y_test):,}")
for c, m in cab_masks.items():
    print(f"  {c}: {m.sum():,} rows")

def metrics(y, yhat):
    yhat = np.clip(yhat, 0, None)
    mae  = float(np.mean(np.abs(y - yhat)))
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2   = float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
    mask = y > 0
    mape = float(np.mean(np.abs((y[mask] - yhat[mask]) / y[mask])) * 100) if mask.sum() else float('nan')
    wmae = float(np.dot(np.abs(y - yhat), y) / y.sum()) if y.sum() > 0 else float('nan')
    return {'MAE': round(mae, 2), 'RMSE': round(rmse, 2),
            'R2':  round(r2, 4),  'MAPE': round(mape, 2), 'WMAE': round(wmae, 2)}

MODELS = [
    ('Baseline',               'Global mean baseline'),
    ('Ridge_-_no_lags,_no_wx', 'Ridge — no lags, no weather'),
    ('Ridge_-_no_lags_p_wx',   'Ridge — no lags + weather'),
    ('Ridge_-_lags,_no_wx',    'Ridge — lags, no weather'),
    ('Ridge_-_full',           'Ridge — full features'),
    ('RF_-_no_lags,_no_wx',    'RF — no lags, no weather'),
    ('RF_-_no_lags_p_wx',      'RF — no lags + weather'),
    ('RF_-_lags,_no_wx',       'RF — lags, no weather'),
    ('RF_-_full',              'RF — full features'),
    ('XGB_-_no_lags,_no_wx',   'XGB — no lags, no weather'),
    ('XGB_-_no_lags_p_wx',     'XGB — no lags + weather'),
    ('XGB_-_lags,_no_wx',      'XGB — lags, no weather'),
    ('XGB_-_full',             'XGB — full features'),
]

file_map = {f.replace('.npy', ''): f
            for f in os.listdir(PRED_DIR) if f.endswith('.npy')}

all_rows = []

for cab in ['fhvhv', 'yellow', 'green', 'all']:
    mask = cab_masks[cab]
    y    = y_test[mask]

    print(f"\n{'='*70}")
    print(f"  Cab type: {cab.upper()}  ({mask.sum():,} rows)")
    print(f"  Mean demand: {y.mean():.2f} trips/hr")
    print(f"{'='*70}")
    print(f"  {'Model':<35} {'MAE':>7} {'RMSE':>7} {'R²':>7} {'MAPE%':>7} {'WMAE':>8}")
    print(f"  {'-'*73}")

    for key, display in MODELS:
        if key not in file_map:
            continue
        yhat = np.load(os.path.join(PRED_DIR, file_map[key]))[mask]
        m = metrics(y, yhat)
        print(f"  {display:<35} {m['MAE']:>7.2f} {m['RMSE']:>7.2f} "
              f"{m['R2']:>7.4f} {m['MAPE']:>7.2f} {m['WMAE']:>8.2f}")
        all_rows.append({'cab_type': cab, 'model': display, **m})

out_df = pd.DataFrame(all_rows)
out_path = os.path.join(OUT_DIR, 'all_metrics_no_fhv_2024_by_cab.csv')
out_df.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")

print(f"{'Cab':<8} {'Config':<22} {'MAE':>7} {'RMSE':>7} {'R²':>7} {'MAPE%':>7} {'WMAE':>8}")
print('-' * 65)
for cab in ['fhvhv', 'yellow', 'green', 'all']:
    for key, display in [('XGB_-_lags,_no_wx', 'lags, no weather'),
                         ('XGB_-_full',         'full features')]:
        mask = cab_masks[cab]
        y    = y_test[mask]
        yhat = np.load(os.path.join(PRED_DIR, file_map[key]))[mask]
        m    = metrics(y, yhat)
        print(f"  {cab:<8} {display:<22} {m['MAE']:>7.2f} {m['RMSE']:>7.2f} "
              f"{m['R2']:>7.4f} {m['MAPE']:>7.2f} {m['WMAE']:>8.2f}")
    print()

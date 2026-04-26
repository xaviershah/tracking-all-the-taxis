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
y_test = df.loc[test_mask, 'trip_count'].values.astype(np.float64)
print(f"  Test rows: {len(y_test):,}")
del df

def mae(y, yhat):
    return float(np.mean(np.abs(y - yhat)))

def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))

def r2(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot)

def mape(y, yhat):
    mask = y > 0
    return float(np.mean(np.abs((y[mask] - yhat[mask]) / y[mask])) * 100)

def wmae(y, yhat):
    yhat = np.clip(yhat, 0, None)
    return float(np.dot(np.abs(y - yhat), y) / y.sum())

MODELS = [
    'Baseline',
    'Ridge_-_no_lags,_no_wx',
    'Ridge_-_no_lags_p_wx',
    'Ridge_-_lags,_no_wx',
    'Ridge_-_full',
    'RF_-_no_lags,_no_wx',
    'RF_-_no_lags_p_wx',
    'RF_-_lags,_no_wx',
    'RF_-_full',
    'XGB_-_no_lags,_no_wx',
    'XGB_-_no_lags_p_wx',
    'XGB_-_lags,_no_wx',
    'XGB_-_full',
]

DISPLAY = {
    'Baseline':               'Global mean baseline',
    'Ridge_-_no_lags,_no_wx': 'Ridge — no lags, no weather',
    'Ridge_-_no_lags_p_wx':   'Ridge — no lags + weather',
    'Ridge_-_lags,_no_wx':    'Ridge — lags, no weather',
    'Ridge_-_full':            'Ridge — full features',
    'RF_-_no_lags,_no_wx':    'RF — no lags, no weather',
    'RF_-_no_lags_p_wx':      'RF — no lags + weather',
    'RF_-_lags,_no_wx':       'RF — lags, no weather',
    'RF_-_full':               'RF — full features',
    'XGB_-_no_lags,_no_wx':   'XGB — no lags, no weather',
    'XGB_-_no_lags_p_wx':     'XGB — no lags + weather',
    'XGB_-_lags,_no_wx':      'XGB — lags, no weather',
    'XGB_-_full':              'XGB — full features',
}

file_map = {f.replace('.npy', ''): f for f in os.listdir(PRED_DIR) if f.endswith('.npy')}

rows = []
print(f"\n{'Model':<35} {'MAE':>8} {'RMSE':>8} {'R²':>7} {'MAPE%':>7} {'WMAE':>8}")
print('-' * 77)

for key in MODELS:
    if key not in file_map:
        print(f"  WARNING: no predictions for {key}")
        continue
    yhat = np.clip(np.load(os.path.join(PRED_DIR, file_map[key])), 0, None)
    m = {
        'model':   DISPLAY[key],
        'MAE':     round(mae(y_test, yhat), 4),
        'RMSE':    round(rmse(y_test, yhat), 4),
        'R2':      round(r2(y_test, yhat), 4),
        'MAPE':    round(mape(y_test, yhat), 2),
        'WMAE':    round(wmae(y_test, yhat), 4),
    }
    rows.append(m)
    print(f"  {DISPLAY[key]:<33} {m['MAE']:>8.2f} {m['RMSE']:>8.2f} "
          f"{m['R2']:>7.4f} {m['MAPE']:>7.2f} {m['WMAE']:>8.2f}")

out_df = pd.DataFrame(rows)
out_path = os.path.join(OUT_DIR, 'all_metrics_no_fhv_2024.csv')
out_df.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")

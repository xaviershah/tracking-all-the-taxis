import os
import numpy as np
import pandas as pd

DATA_PATH = '/home/chongshengwang/ads/all_types_with_fhvhv_dense_filtered_holidays.parquet'
PRED_DIR  = '/home/chongshengwang/ads/short_term_prediction/outputs/model_comparison_no_fhv_2024/dm_predictions'
OUT_DIR   = '/home/chongshengwang/ads/short_term_prediction/outputs/model_comparison_no_fhv_2024'

TEST_MONTHS = [4, 5, 6, 7, 8, 9, 10, 11, 12]
N_BOOT = 10_000
SUB_N  = 100_000
SEED   = 42

df = pd.read_parquet(DATA_PATH, columns=['trip_count', 'year', 'month', 'cab_type'])
df = df[df['cab_type'] != 'fhv'].reset_index(drop=True)
year_arr  = df['year'].to_numpy()
month_arr = df['month'].to_numpy()
test_mask = (year_arr == 2024) & np.isin(month_arr, TEST_MONTHS)
y_test = df.loc[test_mask, 'trip_count'].values.astype(np.float64)
del df
print(f"  Test rows: {len(y_test):,}")

file_map = {f.replace('.npy', ''): f for f in os.listdir(PRED_DIR) if f.endswith('.npy')}

def load_pred(key):
    fname = file_map.get(key)
    if fname is None:
        raise FileNotFoundError(f"No cached predictions for: {key}")
    return np.clip(np.load(os.path.join(PRED_DIR, fname)), 0, None)

PAIRS = [
    ('Ridge_-_no_lags,_no_wx', 'Ridge_-_no_lags_p_wx', 'Ridge — no lags'),
    ('RF_-_no_lags,_no_wx',    'RF_-_no_lags_p_wx',    'RF — no lags'),
    ('XGB_-_no_lags,_no_wx',   'XGB_-_no_lags_p_wx',   'XGB — no lags'),
    ('Ridge_-_lags,_no_wx',    'Ridge_-_full',          'Ridge — lags'),
    ('RF_-_lags,_no_wx',       'RF_-_full',             'RF — lags'),
    ('XGB_-_lags,_no_wx',      'XGB_-_full',            'XGB — lags'),
]

preds = {}
for key_nwx, key_wx, _ in PAIRS:
    for key in [key_nwx, key_wx]:
        if key not in preds:
            preds[key] = load_pred(key)

rng     = np.random.default_rng(SEED)
sub_idx = rng.choice(len(y_test), size=SUB_N, replace=False)
y_sub   = y_test[sub_idx]
pred_sub = {k: preds[k][sub_idx] for k in preds}

def wmae_idx(y, yhat, idx):
    yt = y[idx]; yp = yhat[idx]
    return float(np.dot(np.abs(yt - yp), yt) / yt.sum())

def wmae_sub(k):
    return float(np.dot(np.abs(y_sub - pred_sub[k]), y_sub) / y_sub.sum())

boot_diff = {label: np.zeros(N_BOOT) for _, _, label in PAIRS}

for i in range(N_BOOT):
    idx = rng.integers(0, SUB_N, size=SUB_N)
    for key_nwx, key_wx, label in PAIRS:
        w_nwx = wmae_idx(y_sub, pred_sub[key_nwx], idx)
        w_wx  = wmae_idx(y_sub, pred_sub[key_wx],  idx)
        boot_diff[label][i] = w_nwx - w_wx
    if (i + 1) % 2000 == 0:
        print(f"  {i+1:,} / {N_BOOT:,}")

print(f"  {'Comparison':<22} {'Δ WMAE':>8} {'CI lower':>10} {'CI upper':>10}  Sig")
print(f"  {'-'*60}")

rows = []
for key_nwx, key_wx, label in PAIRS:
    pt = wmae_sub(key_nwx) - wmae_sub(key_wx)
    lo = np.percentile(boot_diff[label], 2.5)
    hi = np.percentile(boot_diff[label], 97.5)
    sig = '***' if (lo > 0 or hi < 0) else '(ns)'
    print(f"  {label:<22} {pt:>+8.3f} {lo:>+10.3f} {hi:>+10.3f}  {sig}")
    rows.append({'comparison': label, 'delta_wmae': round(pt, 4),
                 'ci_lo': round(lo, 4), 'ci_hi': round(hi, 4), 'sig': sig})

out_path = os.path.join(OUT_DIR, 'bootstrap_pairwise_weather_ci.csv')
pd.DataFrame(rows).to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")

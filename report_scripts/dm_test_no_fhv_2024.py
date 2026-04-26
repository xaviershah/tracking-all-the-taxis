import os
import numpy as np
import pandas as pd
import scipy.stats as stats

DATA_PATH = '/home/chongshengwang/ads/all_types_with_fhvhv_dense_filtered_holidays.parquet'
PRED_DIR  = '/home/chongshengwang/ads/short_term_prediction/outputs/model_comparison_no_fhv_2024/dm_predictions'
OUT_DIR   = '/home/chongshengwang/ads/short_term_prediction/outputs/model_comparison_no_fhv_2024'

TARGET      = 'trip_count'
TRAIN_YEARS = [2022, 2023]
TEST_MONTHS = [4, 5, 6, 7, 8, 9, 10, 11, 12]

LAG_MODELS = {
    'Ridge — lags, no wx', 'Ridge — full',
    'RF — lags, no wx',    'RF — full',
    'XGB — lags, no wx',   'XGB — full',
}

def newey_west_var(d, max_lag=None):
    T = len(d)
    if max_lag is None:
        max_lag = int(np.floor(4 * (T / 100) ** (2 / 9)))
    d_dm = d - d.mean()
    lrv = np.dot(d_dm, d_dm) / T
    for k in range(1, max_lag + 1):
        lrv += 2 * (1 - k / (max_lag + 1)) * np.dot(d_dm[k:], d_dm[:-k]) / T
    return max(lrv, 1e-12)

def dm_test(l1, l2):
    d = l1 - l2; T = len(d); h = 1
    cf = np.sqrt((T + 1 - 2*h + h*(h-1)/T) / T)
    dm_stat = cf * d.mean() / np.sqrt(newey_west_var(d) / T)
    p_val = 2 * stats.t.sf(abs(dm_stat), df=T-1)
    return float(dm_stat), float(p_val)

def pred_path(label):
    return os.path.join(PRED_DIR,
        label.replace(' ', '_').replace('—', '-').replace('+', 'p') + '.npy')

def get_preds(label):
    p = pred_path(label)
    if not os.path.exists(p):
        raise FileNotFoundError(f"No cached predictions for: {label}")
    return np.load(p)

df = pd.read_parquet(DATA_PATH, columns=[TARGET, 'year', 'month', 'cab_type',
                                          'lag_168', 'hour', 'day_of_week'])
df = df[df['cab_type'] != 'fhv'].reset_index(drop=True)

year_arr  = df['year'].to_numpy()
month_arr = df['month'].to_numpy()
lag_ok    = ~np.isnan(df['lag_168'].to_numpy().astype(float))
test_mask     = (year_arr == 2024) & np.isin(month_arr, TEST_MONTHS)
test_lag_mask = test_mask & lag_ok

y_test     = df.loc[test_mask,     TARGET].values.astype(np.float64)
y_test_lag = df.loc[test_lag_mask, TARGET].values.astype(np.float64)

hour_idx_all = (df['month'].to_numpy().astype(int) * 31 * 24
                + df['day_of_week'].to_numpy().astype(int) * 24
                + df['hour'].to_numpy().astype(int))
h_test_all = hour_idx_all[test_mask]
h_test_lag = hour_idx_all[test_lag_mask]
del df

MODELS = [
    'Baseline',
    'Ridge — no lags, no wx', 'Ridge — no lags + wx',
    'Ridge — lags, no wx',    'Ridge — full',
    'RF — no lags, no wx',    'RF — no lags + wx',
    'RF — lags, no wx',       'RF — full',
    'XGB — no lags, no wx',   'XGB — no lags + wx',
    'XGB — lags, no wx',      'XGB — full',
]

loss_series = {}
for label in MODELS:
    preds = get_preds(label)
    uses_lags = label in LAG_MODELS
    y    = y_test_lag if uses_lags else y_test
    hidx = h_test_lag if uses_lags else h_test_all
    l = y * np.abs(y - preds)
    df_h = pd.DataFrame({'hour': hidx, 'loss': l})
    loss_series[label] = df_h.groupby('hour')['loss'].mean().sort_index()
    print(f"  {label}: {len(loss_series[label])} hourly steps")

DM_PAIRS = [
    ('Ridge — no lags, no wx', 'Ridge — no lags + wx',  'Ridge: weather? (no lags)'),
    ('RF — no lags, no wx',    'RF — no lags + wx',     'RF: weather? (no lags)'),
    ('XGB — no lags, no wx',   'XGB — no lags + wx',    'XGB: weather? (no lags)'),
    ('Ridge — lags, no wx',    'Ridge — full',           'Ridge: weather? (lags)'),
    ('RF — lags, no wx',       'RF — full',              'RF: weather? (lags)'),
    ('XGB — lags, no wx',      'XGB — full',             'XGB: weather? (lags)'),
    ('Ridge — no lags, no wx', 'Ridge — lags, no wx',   'Ridge: lags? (no wx)'),
    ('RF — no lags, no wx',    'RF — lags, no wx',      'RF: lags? (no wx)'),
    ('XGB — no lags, no wx',   'XGB — lags, no wx',     'XGB: lags? (no wx)'),
    ('Ridge — full',           'RF — full',              'Full: Ridge vs RF'),
    ('RF — full',              'XGB — full',             'Full: RF vs XGB'),
    ('Ridge — full',           'XGB — full',             'Full: Ridge vs XGB'),
]

dm_results = []
for m1, m2, desc in DM_PAIRS:
    s1, s2 = loss_series[m1], loss_series[m2]
    common = s1.index.intersection(s2.index)
    dm_stat, p_val = dm_test(s1.loc[common].values, s2.loc[common].values)
    sig = ('***' if p_val < 0.001 else '**' if p_val < 0.01 else
           '*' if p_val < 0.05 else '(ns)')
    direction = 'm1 worse' if dm_stat > 0 else 'm2 worse'
    print(f"  {desc:<40}  DM={dm_stat:+.3f}  p={p_val:.4f} {sig}  [{direction}]")
    dm_results.append({'comparison': desc, 'model_1': m1, 'model_2': m2,
                       'DM_stat': round(dm_stat, 4), 'p_value': round(p_val, 6),
                       'significant': sig, 'T_hours': len(common)})

out_path = os.path.join(OUT_DIR, 'dm_results_no_fhv_2024.csv')
pd.DataFrame(dm_results).to_csv(out_path, index=False)
print(f"\nDM results saved to {out_path}")
print("Done.")

import os
import numpy as np
import pandas as pd
import matplotlib


matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

DATA_PATH = '/home/chongshengwang/ads/all_types_with_fhvhv_dense_filtered_holidays.parquet'

PRED_DIR  = '/home/chongshengwang/ads/short_term_prediction/outputs/model_comparison_no_fhv_2024/dm_predictions'
OUT_DIR   = '/home/chongshengwang/ads/short_term_prediction/outputs/model_comparison_no_fhv_2024'

TARGET      = 'trip_count'
TEST_MONTHS = [4, 5, 6, 7, 8, 9, 10, 11, 12]
N_BOOT      = 10_000
SUB_N       = 100_000
SEED        = 42

MODELS = [
    'Baseline',
    'Ridge — no lags, no wx', 'Ridge — no lags + wx',
    'Ridge — lags, no wx',    'Ridge — full',
    'RF — no lags, no wx',    'RF — no lags + wx',
    'RF — lags, no wx',       'RF — full',
    'XGB — no lags, no wx',   'XGB — no lags + wx',
    'XGB — lags, no wx',      'XGB — full',
]

COLORS = {
    'Baseline':               '#95A5A6',
    'Ridge — no lags, no wx': '#2E86C1', 'Ridge — no lags + wx': '#7FB3D3',
    'Ridge — lags, no wx':    '#2E86C1', 'Ridge — full':          '#7FB3D3',
    'RF — no lags, no wx':    '#1E8449', 'RF — no lags + wx':     '#76B041',
    'RF — lags, no wx':       '#1E8449', 'RF — full':              '#76B041',
    'XGB — no lags, no wx':   '#CA6F1E', 'XGB — no lags + wx':    '#F0A500',
    'XGB — lags, no wx':      '#CA6F1E', 'XGB — full':             '#F0A500',
}

DISPLAY = {
    'Baseline':               'Global mean baseline',
    'Ridge — no lags, no wx': 'Ridge — no lags, no weather',
    'Ridge — no lags + wx':   'Ridge — no lags + weather',
    'Ridge — lags, no wx':    'Ridge — lags, no weather',
    'Ridge — full':            'Ridge — full features',
    'RF — no lags, no wx':    'RF — no lags, no weather',
    'RF — no lags + wx':      'RF — no lags + weather',
    'RF — lags, no wx':       'RF — lags, no weather',
    'RF — full':               'RF — full features',
    'XGB — no lags, no wx':   'XGB — no lags, no weather',
    'XGB — no lags + wx':     'XGB — no lags + weather',
    'XGB — lags, no wx':      'XGB — lags, no weather',
    'XGB — full':              'XGB — full features',
}

NO_LAG_KEYS = ['Baseline',
                'Ridge — no lags, no wx', 'Ridge — no lags + wx',
                'RF — no lags, no wx',    'RF — no lags + wx',
                'XGB — no lags, no wx',   'XGB — no lags + wx']
LAG_KEYS    = ['Ridge — lags, no wx', 'Ridge — full',
               'RF — lags, no wx',    'RF — full',
               'XGB — lags, no wx',   'XGB — full']

print("Loading test actuals...")
df = pd.read_parquet(DATA_PATH, columns=[TARGET, 'year', 'month', 'cab_type'])
df = df[df['cab_type'] != 'fhv'].reset_index(drop=True)
year_arr  = df['year'].to_numpy()
month_arr = df['month'].to_numpy()
test_mask = (year_arr == 2024) & np.isin(month_arr, TEST_MONTHS)
y_test = df.loc[test_mask, TARGET].values.astype(np.float64)
del df

def pred_path(label):
    return os.path.join(PRED_DIR,
        label.replace(' ', '_').replace('—', '-').replace('+', 'p') + '.npy')

preds = {}
for label in MODELS:
    p = pred_path(label)
    if not os.path.exists(p):
        print(f"  WARNING: no cached predictions for {label}")
        continue
    preds[label] = np.clip(np.load(p), 0, None)
loaded_keys = list(preds.keys())

rng     = np.random.default_rng(SEED)
sub_idx = rng.choice(len(y_test), size=SUB_N, replace=False)
y_sub   = y_test[sub_idx]
pred_sub = {k: preds[k][sub_idx] for k in loaded_keys}

def wmae_idx(y, yhat, idx):
    yt = y[idx]; yp = yhat[idx]
    return float(np.dot(np.abs(yt - yp), yt) / yt.sum())

def wmae_sub(k):
    return float(np.dot(np.abs(y_sub - pred_sub[k]), y_sub) / y_sub.sum())
boot = {k: np.zeros(N_BOOT) for k in loaded_keys}
for i in range(N_BOOT):
    idx = rng.integers(0, SUB_N, size=SUB_N)
    for k in loaded_keys:
        boot[k][i] = wmae_idx(y_sub, pred_sub[k], idx)
    if (i + 1) % 2000 == 0:
        print(f"  {i+1:,} / {N_BOOT:,}")

ci_rows = []
for k in loaded_keys:
    pt = wmae_sub(k)
    lo = np.percentile(boot[k], 2.5)
    hi = np.percentile(boot[k], 97.5)
    ci_rows.append({'key': k, 'wmae': pt, 'ci_lo': lo, 'ci_hi': hi})
    print(f"  {k:<30}  WMAE={pt:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

pd.DataFrame(ci_rows).to_csv(
    os.path.join(OUT_DIR, 'bootstrap_wmae_no_fhv_2024.csv'), index=False)

DIFF_PAIRS = [
    ('XGB — no lags, no wx', 'XGB — no lags + wx', 'nolag'),
    ('XGB — lags, no wx',    'XGB — full',          'lag'),
]
diffs = {}
for m1, m2, tag in DIFF_PAIRS:
    if m1 not in boot or m2 not in boot:
        continue
    diff_dist  = boot[m1] - boot[m2]
    diff_point = wmae_sub(m1) - wmae_sub(m2)
    lo = np.percentile(diff_dist, 2.5)
    hi = np.percentile(diff_dist, 97.5)
    diffs[tag] = (diff_point, lo, hi)
    sig = '***' if not (lo <= 0 <= hi) else '(ns)'
    print(f"  XGB weather diff ({tag}): Δ={diff_point:+.4f}  [{lo:+.4f}, {hi:+.4f}]  {sig}")

ci_lookup = {r['key']: r for r in ci_rows}

def draw_panel(ax, keys, title, diff_tag):
    keys   = [k for k in keys if k in ci_lookup]
    pts    = [ci_lookup[k]['wmae']  for k in keys]
    los    = [ci_lookup[k]['ci_lo'] for k in keys]
    his    = [ci_lookup[k]['ci_hi'] for k in keys]
    colors = [COLORS[k]             for k in keys]
    labels = [DISPLAY[k]            for k in keys]

    ax.barh(range(len(keys)), pts,
            xerr=[[p - l for p, l in zip(pts, los)],
                  [h - p for p, h in zip(pts, his)]],
            color=colors, edgecolor='none', alpha=0.85,
            capsize=4, height=0.55,
            error_kw={'elinewidth': 1.3, 'ecolor': '#2C3E50', 'capthick': 1.3})
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel('Weighted MAE (lower = better)', fontsize=9)
    ax.set_title(title, fontsize=10, pad=8)
    ax.tick_params(axis='y', length=0)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis='x', linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)

    x_range = max(his) - min(los)
    x_right = max(his) + x_range * 0.38
    for i, (pt, lo, hi) in enumerate(zip(pts, los, his)):
        ax.text(hi + x_range * 0.015, i,
                f'{pt:.2f}  [{lo:.2f}, {hi:.2f}]',
                va='center', ha='left', fontsize=7.5,
                color='#2C3E50', fontfamily='monospace')
    ax.set_xlim(min(los) - x_range * 0.04, x_right)

    if diff_tag in diffs:
        dp, dlo, dhi = diffs[diff_tag]
        sig = '***' if not (dlo <= 0 <= dhi) else '(ns)'
        lbl = 'XGB: no-wx vs +wx' if diff_tag == 'nolag' else 'XGB: lags no-wx vs full'
        ax.text(0.98, 0.02,
                f'{lbl}\nΔ = {dp:+.2f}  95% CI [{dlo:+.2f}, {dhi:+.2f}]  {sig}',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=7.5,
                bbox=dict(boxstyle='round,pad=0.35', facecolor='#FDFEFE',
                          edgecolor='#AEB6BF', linewidth=0.8))

fig, axes = plt.subplots(1, 2, figsize=(18, 5.5), constrained_layout=True)
draw_panel(axes[0], NO_LAG_KEYS,
           'Without Lag Features\n95% Bootstrap CI on Weighted MAE', 'nolag')
draw_panel(axes[1], LAG_KEYS,
           'With Lag Features\n95% Bootstrap CI on Weighted MAE', 'lag')
fig.suptitle('Bootstrap Confidence Intervals — Volume-Weighted MAE\n'
             f'NYC Taxi  |  FHV removed  |  Test: Apr–Dec 2024  |  '
             f'n={N_BOOT:,} resamples, subsample={SUB_N:,}', fontsize=11)
ci_png = os.path.join(OUT_DIR, 'bootstrap_wmae_no_fhv_2024_ci.png')
plt.savefig(ci_png, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nCI plot saved to {ci_png}")

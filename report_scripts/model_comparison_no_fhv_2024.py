import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DATA_PATH = '/home/chongshengwang/ads/all_types_with_fhvhv_dense_filtered_holidays.parquet'
PRED_DIR  = '/home/chongshengwang/ads/short_term_prediction/outputs/model_comparison_no_fhv_2024/dm_predictions'
OUT_DIR   = '/home/chongshengwang/ads/short_term_prediction/outputs/model_comparison_no_fhv_2024'

TARGET      = 'trip_count'
LAG_FEATURES = ['lag_1', 'lag_2', 'lag_24', 'lag_168']
TRAIN_YEARS = [2022, 2023]
TEST_MONTHS = [4, 5, 6, 7, 8, 9, 10, 11, 12]

MODELS_USES_LAGS = {
    'Baseline':               False,
    'Ridge — no lags, no wx': False,
    'Ridge — no lags + wx':   False,
    'Ridge — lags, no wx':    True,
    'Ridge — full':            True,
    'RF — no lags, no wx':    False,
    'RF — no lags + wx':      False,
    'RF — lags, no wx':       True,
    'RF — full':               True,
    'XGB — no lags, no wx':   False,
    'XGB — no lags + wx':     False,
    'XGB — lags, no wx':      True,
    'XGB — full':              True,
}

def weighted_mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float64), 0, None)
    return float(np.dot(np.abs(y_true - y_pred), y_true) / y_true.sum())

def pred_path(label):
    return os.path.join(PRED_DIR,
        label.replace(' ', '_').replace('—', '-').replace('+', 'p') + '.npy')

df = pd.read_parquet(DATA_PATH, columns=[TARGET, 'year', 'month', 'cab_type',
                                          'lag_168']).reset_index(drop=True)
df = df[df['cab_type'] != 'fhv'].reset_index(drop=True)

year_arr  = df['year'].to_numpy()
month_arr = df['month'].to_numpy()
lag_ok    = ~np.isnan(df['lag_168'].to_numpy().astype(float))
test_mask     = (year_arr == 2024) & np.isin(month_arr, TEST_MONTHS)
test_lag_mask = test_mask & lag_ok

y_test     = df.loc[test_mask,     TARGET].values.astype(np.float64)
y_test_lag = df.loc[test_lag_mask, TARGET].values.astype(np.float64)

wmae_results = {}
for label, uses_lags in MODELS_USES_LAGS.items():
    p = pred_path(label)
    if not os.path.exists(p):
        print(f"  WARNING: no cached predictions for {label}")
        continue
    preds = np.load(p)
    y_ref = y_test_lag if uses_lags else y_test
    wmae_results[label] = weighted_mae(y_ref, preds)

print(f"\n{'Model':<35} {'WMAE':>8}")
print('-' * 45)
for k, v in wmae_results.items():
    print(f"  {k:<33} {v:.4f}")

pd.DataFrame([{'model': k, 'wmae': v} for k, v in wmae_results.items()]).to_csv(
    os.path.join(OUT_DIR, 'model_comparison_no_fhv_2024_wmae.csv'), index=False)

plot_groups = [
    {'title': 'No lags, no weather', 'subtitle': '(time + zone only)',
     'entries': [('Baseline',             'Mean\nbaseline', '#95A5A6'),
                 ('Ridge — no lags, no wx', 'Ridge',   '#3498DB'),
                 ('RF — no lags, no wx',    'RF',      '#27AE60'),
                 ('XGB — no lags, no wx',   'XGBoost', '#E67E22')]},
    {'title': 'No lags + weather', 'subtitle': '',
     'entries': [('Ridge — no lags + wx', 'Ridge',   '#3498DB'),
                 ('RF — no lags + wx',    'RF',      '#27AE60'),
                 ('XGB — no lags + wx',   'XGBoost', '#E67E22')]},
    {'title': 'With lags, no weather', 'subtitle': '',
     'entries': [('Ridge — lags, no wx', 'Ridge',   '#3498DB'),
                 ('RF — lags, no wx',    'RF',      '#27AE60'),
                 ('XGB — lags, no wx',   'XGBoost', '#E67E22')]},
    {'title': 'Full features', 'subtitle': '(lags + weather)',
     'entries': [('Ridge — full', 'Ridge',   '#3498DB'),
                 ('RF — full',    'RF',      '#27AE60'),
                 ('XGB — full',   'XGBoost', '#E67E22')]},
]

fig, axes = plt.subplots(1, 4, figsize=(14, 6))
fig.subplots_adjust(bottom=0.22, top=0.85, wspace=0.38)
for ax, group in zip(axes, plot_groups):
    entries = [(k, lbl, col) for k, lbl, col in group['entries'] if k in wmae_results]
    vals = [wmae_results[k] for k, _, _ in entries]
    bars = ax.bar(range(len(entries)), vals,
                  color=[col for _, _, col in entries], alpha=0.88,
                  edgecolor='white', linewidth=0.5, width=0.55)
    subtitle = f"\n{group['subtitle']}" if group['subtitle'] else ''
    ax.set_title(f"{group['title']}{subtitle}", fontsize=9, pad=6)
    ax.set_xticks(range(len(entries)))
    ax.set_xticklabels([lbl for _, lbl, _ in entries], fontsize=9, ha='center')
    ax.set_ylabel('Weighted MAE (lower = better)', fontsize=8)
    ax.tick_params(axis='x', length=0)
    if vals:
        ax.set_ylim(0, max(vals) * 1.16)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.018,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

legend_els = [mpatches.Patch(facecolor=c, label=l) for c, l in
              [('#95A5A6','Baseline'), ('#3498DB','Ridge'),
               ('#27AE60','Random Forest'), ('#E67E22','XGBoost')]]
fig.legend(handles=legend_els, loc='lower center', ncol=4,
           fontsize=9, bbox_to_anchor=(0.5, 0.01), frameon=False)
fig.suptitle('Model Comparison: Ridge vs RF vs XGBoost\n'
             'Volume-Weighted MAE  |  Test: Apr–Dec 2024  |  FHV removed', fontsize=11)
out_png = os.path.join(OUT_DIR, 'model_comparison_no_fhv_2024.png')
plt.savefig(out_png, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nChart saved to {out_png}")

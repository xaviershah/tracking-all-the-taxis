import gc
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = '/home/chongshengwang/ads/all_types_with_fhvhv_dense_filtered_holidays.parquet'
XGB_DIR   = '/home/chongshengwang/ads/short_term_prediction/outputs/ablation_no_fhv_2024'
MC_DIR    = '/home/chongshengwang/ads/short_term_prediction/outputs/model_comparison_no_fhv_2024'
PRED_DIR  = os.path.join(MC_DIR, 'dm_predictions')
os.makedirs(PRED_DIR, exist_ok=True)

TARGET           = 'trip_count'
CAT_FEATURES     = ['cab_type']
ZONE_FEATURES    = ['PULocationID']
TIME_FEATURES    = ['hour', 'day_of_week', 'month', 'is_weekend',
                    'is_rush_hour', 'is_public_holiday']
WEATHER_FEATURES = ['precipitation', 'temperature', 'wind_speed',
                    'visibility', 'rain_category']
LAG_FEATURES     = ['lag_1', 'lag_2', 'lag_24', 'lag_168']
ALL_FEATURES     = (CAT_FEATURES + ZONE_FEATURES + TIME_FEATURES
                    + WEATHER_FEATURES + LAG_FEATURES)
OHE_COLS         = ['cab_type', 'rain_category']

TRAIN_YEARS = [2022, 2023]
TEST_MONTHS = [4, 5, 6, 7, 8, 9, 10, 11, 12]
SEED        = 42

XGB_FEAT_MAP = {
    'A_full':              ALL_FEATURES,
    'B_no_weather':        CAT_FEATURES + ZONE_FEATURES + TIME_FEATURES + LAG_FEATURES,
    'D_no_lag_weather':    CAT_FEATURES + ZONE_FEATURES + TIME_FEATURES + WEATHER_FEATURES,
    'E_no_lag_no_weather': CAT_FEATURES + ZONE_FEATURES + TIME_FEATURES,
}

def weighted_mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float64), 0, None)
    return float(np.dot(np.abs(y_true - y_pred), y_true) / y_true.sum())

def make_pipeline(features, estimator):
    cat_cols = [f for f in OHE_COLS if f in features]
    num_cols = [f for f in features if f not in OHE_COLS]
    transformers = []
    if cat_cols:
        transformers.append(('ohe', OneHotEncoder(handle_unknown='ignore',
                                                   sparse_output=True), cat_cols))
    if num_cols:
        transformers.append(('scaler', StandardScaler(), num_cols))
    return Pipeline([('prep', ColumnTransformer(transformers=transformers,
                                                remainder='drop')),
                     ('model', estimator)])

def pred_path(label):
    return os.path.join(PRED_DIR,
        label.replace(' ', '_').replace('—', '-').replace('+', 'p') + '.npy')

def get_preds(label):
    p = pred_path(label)
    return np.load(p) if os.path.exists(p) else None

def save_preds(label, arr):
    np.save(pred_path(label), arr)

load_cols = list(dict.fromkeys(ALL_FEATURES + [TARGET, 'year', 'month']))
df = pd.read_parquet(DATA_PATH, columns=load_cols).reset_index(drop=True)
df = df[df['cab_type'] != 'fhv'].copy().reset_index(drop=True)

for col in ['precipitation', 'temperature', 'wind_speed', 'visibility',
            'lag_1', 'lag_2', 'lag_24', 'lag_168', TARGET]:
    df[col] = df[col].astype(np.float32)
for col in ['hour', 'day_of_week', 'month', 'is_weekend',
            'is_rush_hour', 'is_public_holiday']:
    df[col] = df[col].astype(np.int8)
df['PULocationID'] = df['PULocationID'].astype(np.int16)
df['cab_type'] = df['cab_type'].astype('category')

year_arr  = df['year'].to_numpy()
month_arr = df['month'].to_numpy()
lag_ok    = ~np.isnan(df['lag_168'].to_numpy().astype(float))

train_mask     = np.isin(year_arr, TRAIN_YEARS)
test_mask      = (year_arr == 2024) & np.isin(month_arr, TEST_MONTHS)
train_lag_mask = train_mask & lag_ok
test_lag_mask  = test_mask  & lag_ok

y_test     = df.loc[test_mask,     TARGET].values.astype(np.float64)
y_test_lag = df.loc[test_lag_mask, TARGET].values.astype(np.float64)

MODELS = {
    'Baseline':               None,
    'Ridge — no lags, no wx': ('ridge', False, CAT_FEATURES + ZONE_FEATURES + TIME_FEATURES),
    'Ridge — no lags + wx':   ('ridge', False, CAT_FEATURES + ZONE_FEATURES + TIME_FEATURES + WEATHER_FEATURES),
    'Ridge — lags, no wx':    ('ridge', True,  CAT_FEATURES + ZONE_FEATURES + TIME_FEATURES + LAG_FEATURES),
    'Ridge — full':            ('ridge', True,  ALL_FEATURES),
    'RF — no lags, no wx':    ('rf',    False, CAT_FEATURES + ZONE_FEATURES + TIME_FEATURES),
    'RF — no lags + wx':      ('rf',    False, CAT_FEATURES + ZONE_FEATURES + TIME_FEATURES + WEATHER_FEATURES),
    'RF — lags, no wx':       ('rf',    True,  CAT_FEATURES + ZONE_FEATURES + TIME_FEATURES + LAG_FEATURES),
    'RF — full':               ('rf',    True,  ALL_FEATURES),
    'XGB — no lags, no wx':   ('xgb',  False, 'E_no_lag_no_weather'),
    'XGB — no lags + wx':     ('xgb',  False, 'D_no_lag_weather'),
    'XGB — lags, no wx':      ('xgb',  True,  'B_no_weather'),
    'XGB — full':              ('xgb',  True,  'A_full'),
}

wmae_results = {}

for label, spec in MODELS.items():
    cached = get_preds(label)
    if cached is not None:
        print(f"  {label}: cached")
        uses_lags = spec is not None and spec[1]
        wmae_results[label] = weighted_mae(
            y_test_lag if uses_lags else y_test, cached)
        continue


    if spec is None:
        global_mean = float(df.loc[train_mask, TARGET].mean())
        preds = np.full(test_mask.sum(), global_mean)
        save_preds(label, preds)
        wmae_results[label] = weighted_mae(y_test, preds)
        gc.collect(); continue

    family, uses_lags, feat_spec = spec

    if family == 'xgb':
        mask     = test_lag_mask if uses_lags else test_mask
        features = XGB_FEAT_MAP[feat_spec]
        X_test   = df.loc[mask, features].copy()
        X_test['cab_type'] = X_test['cab_type'].astype('category')
        model = xgb.XGBRegressor()
        model.load_model(os.path.join(XGB_DIR, f'{feat_spec}_model.ubj'))
        preds = np.clip(model.predict(X_test), 0, None)
        del X_test, model
    else:
        tr_mask  = train_lag_mask if uses_lags else train_mask
        te_mask  = test_lag_mask  if uses_lags else test_mask
        features = feat_spec

        X_train = df.loc[tr_mask, features].copy()
        y_train = df.loc[tr_mask, TARGET].values
        X_test  = df.loc[te_mask, features].copy()
        for col in OHE_COLS:
            if col in features:
                X_train[col] = X_train[col].astype(str)
                X_test[col]  = X_test[col].astype(str)

        if family == 'ridge':
            est = Ridge(alpha=1.0)
        else:
            est = RandomForestRegressor(
                n_estimators=100, max_depth=15, min_samples_leaf=50,
                max_features=0.33, max_samples=500_000,
                n_jobs=-1, random_state=SEED)

        pipe = make_pipeline(features, est)
        pipe.fit(X_train, y_train)
        del X_train, y_train; gc.collect()
        preds = np.clip(pipe.predict(X_test), 0, None)
        del X_test, pipe; gc.collect()

    save_preds(label, preds)
    y_ref = y_test_lag if uses_lags else y_test
    wmae_results[label] = weighted_mae(y_ref, preds)
    print(f"    WMAE = {wmae_results[label]:.4f}  ({len(preds):,} predictions)")
    gc.collect()

del df; gc.collect()

for k, v in wmae_results.items():
    print(f"  {k:<40} {v:.4f}")

pd.DataFrame([{'model': k, 'wmae': v} for k, v in wmae_results.items()]).to_csv(
    os.path.join(MC_DIR, 'model_comparison_no_fhv_2024_wmae.csv'), index=False)

plot_groups = [
    {'title': 'No lags, no weather', 'subtitle': '(time + zone only)',
     'entries': [('Baseline',               'Mean\nbaseline', '#95A5A6'),
                 ('Ridge — no lags, no wx', 'Ridge',          '#3498DB'),
                 ('RF — no lags, no wx',    'RF',             '#27AE60'),
                 ('XGB — no lags, no wx',   'XGBoost',        '#E67E22')]},
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
plt.savefig(os.path.join(MC_DIR, 'model_comparison_no_fhv_2024.png'),
            dpi=150, bbox_inches='tight')
plt.close()

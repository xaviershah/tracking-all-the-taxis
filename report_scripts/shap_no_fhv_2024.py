import gc
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

DATA_PATH = '/home/chongshengwang/ads/all_types_with_fhvhv_dense_filtered_holidays.parquet'
XGB_DIR   = '/home/chongshengwang/ads/short_term_prediction/outputs/ablation_no_fhv_2024'
SHAP_DIR  = '/home/chongshengwang/ads/short_term_prediction/outputs/shap_no_fhv_2024'
os.makedirs(SHAP_DIR, exist_ok=True)

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

TRAIN_YEARS = [2022, 2023]
TEST_MONTHS = [4, 5, 6, 7, 8, 9, 10, 11, 12]
SAMPLE_N    = 10_000
SEED        = 42

FEAT_LABELS = {
    'lag_1': 'lag_1 (1hr)', 'lag_168': 'lag_168 (1wk)',
    'lag_24': 'lag_24 (24hr)', 'lag_2': 'lag_2 (2hr)',
    'hour': 'hour', 'PULocationID': 'zone', 'cab_type': 'cab_type',
    'day_of_week': 'day_of_week', 'month': 'month',
    'is_weekend': 'is_weekend', 'is_rush_hour': 'is_rush_hour',
    'is_public_holiday': 'is_public_holiday', 'precipitation': 'precipitation',
    'temperature': 'temperature', 'wind_speed': 'wind_speed',
    'rain_category': 'rain_category', 'visibility': 'visibility',
}

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
lag_arr   = df['lag_168'].to_numpy()
lag_ok    = ~np.isnan(lag_arr.astype(float))

train_mask     = np.isin(year_arr, TRAIN_YEARS)
test_mask      = (year_arr == 2024) & np.isin(month_arr, TEST_MONTHS)
test_lag_mask  = test_mask & lag_ok

def beeswarm_plot(sv, X_samp, feat_names, title, out_path):
    mean_abs = np.abs(sv).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1]
    sv_s     = sv[:, order]
    X_s      = X_samp.iloc[:, order].copy()
    X_s.columns = [FEAT_LABELS[feat_names[i]] for i in order]
    total   = np.abs(sv).sum()
    lag_pct = np.abs(sv[:, [feat_names.index(f) for f in LAG_FEATURES]]).sum() / total * 100
    wx_pct  = np.abs(sv[:, [feat_names.index(f) for f in WEATHER_FEATURES]]).sum() / total * 100
    print(f"  Lag total: {lag_pct:.1f}%  |  Weather total: {wx_pct:.1f}%")
    for i in order:
        tag = ' <- weather' if feat_names[i] in WEATHER_FEATURES else ''
        print(f"    {FEAT_LABELS[feat_names[i]]:<25} {mean_abs[i]:>8.4f}{tag}")
    fig, ax = plt.subplots(figsize=(9, 7))
    shap.summary_plot(sv_s, X_s, show=False, plot_size=None,
                      color_bar_label='Feature value')
    plt.title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {out_path}")
    return lag_pct, wx_pct

rng = np.random.default_rng(SEED)

df_xgb = df.copy()
df_xgb['cab_type'] = df_xgb['cab_type'].astype('category')
test_xgb = df_xgb.loc[test_lag_mask, ALL_FEATURES].reset_index(drop=True)
idx = rng.choice(len(test_xgb), size=min(SAMPLE_N, len(test_xgb)), replace=False)
X_samp_xgb = test_xgb.iloc[idx].reset_index(drop=True)
del test_xgb, df_xgb; gc.collect()

model_xgb = xgb.XGBRegressor()
model_xgb.load_model(os.path.join(XGB_DIR, 'A_full_model.ubj'))
dm_mat   = xgb.DMatrix(X_samp_xgb, enable_categorical=True)
contribs = model_xgb.get_booster().predict(dm_mat, pred_contribs=True)
sv_xgb   = contribs[:, :-1]
del model_xgb, dm_mat, contribs; gc.collect()

xgb_lag_pct, xgb_wx_pct = beeswarm_plot(
    sv_xgb, X_samp_xgb, ALL_FEATURES,
    f'SHAP Summary — XGBoost Full Model\nTest: Apr–Dec 2024 (no FHV), n={SAMPLE_N:,} sample',
    os.path.join(SHAP_DIR, 'xgb_shap_beeswarm_no_fhv_2024.png'))
del sv_xgb, X_samp_xgb; gc.collect()

df_rf = df.copy()
le_cab  = LabelEncoder()
le_rain = LabelEncoder()
df_rf['cab_type']      = le_cab.fit_transform(df_rf['cab_type'].astype(str)).astype(np.int8)
df_rf['rain_category'] = le_rain.fit_transform(df_rf['rain_category'].astype(str)).astype(np.int8)

X_train_rf = df_rf.loc[train_mask, ALL_FEATURES]
y_train_rf = df_rf.loc[train_mask, TARGET].values
test_rf    = df_rf.loc[test_lag_mask, ALL_FEATURES].reset_index(drop=True)
del df, df_rf; gc.collect()

rf = RandomForestRegressor(
    n_estimators=100, max_depth=15, min_samples_leaf=50,
    max_features=0.33, max_samples=500_000,
    n_jobs=-1, random_state=SEED)
rf.fit(X_train_rf, y_train_rf)
del X_train_rf, y_train_rf; gc.collect()

idx = rng.choice(len(test_rf), size=min(SAMPLE_N, len(test_rf)), replace=False)
X_samp_rf = test_rf.iloc[idx].reset_index(drop=True)
del test_rf; gc.collect()

explainer = shap.TreeExplainer(rf)
sv_rf = np.array(explainer.shap_values(X_samp_rf, check_additivity=False))
del rf, explainer; gc.collect()

rf_lag_pct, rf_wx_pct = beeswarm_plot(
    sv_rf, X_samp_rf, ALL_FEATURES,
    f'SHAP Summary — RF Full Model\nTest: Apr–Dec 2024 (no FHV), n={SAMPLE_N:,} sample',
    os.path.join(SHAP_DIR, 'rf_shap_beeswarm_no_fhv_2024.png'))

fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(2); w = 0.25
lag_pcts   = [xgb_lag_pct, rf_lag_pct]
wx_pcts    = [xgb_wx_pct,  rf_wx_pct]
other_pcts = [100 - l - wx for l, wx in zip(lag_pcts, wx_pcts)]
ax.bar(x - w, lag_pcts,   width=w, color='#4878d0', label='Lag features')
ax.bar(x,     wx_pcts,    width=w, color='#e07b39', label='Weather features')
ax.bar(x + w, other_pcts, width=w, color='#7d7d7d', label='Other')
for i, (l, wx, ot) in enumerate(zip(lag_pcts, wx_pcts, other_pcts)):
    ax.text(i - w, l + 0.5,  f'{l:.1f}%',  ha='center', va='bottom', fontsize=9)
    ax.text(i,     wx + 0.5, f'{wx:.1f}%', ha='center', va='bottom', fontsize=9)
    ax.text(i + w, ot + 0.5, f'{ot:.1f}%', ha='center', va='bottom', fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(['XGBoost', 'Random Forest'], fontsize=11)
ax.set_ylabel('% of total |SHAP|', fontsize=10)
ax.set_title('Feature Group SHAP Contribution\n(no FHV, Test: Apr–Dec 2024)', fontsize=11)
ax.legend(fontsize=9)
ax.set_ylim(0, max(lag_pcts) * 1.12)
plt.tight_layout()
out_grp = os.path.join(SHAP_DIR, 'shap_group_contribution_no_fhv_2024.png')
plt.savefig(out_grp, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nGroup contribution saved to {out_grp}")

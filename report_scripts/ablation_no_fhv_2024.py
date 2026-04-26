
import gc
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

DATA_PATH = '/home/chongshengwang/ads/all_types_with_fhvhv_dense_filtered_holidays.parquet'
OUT_DIR   = '/home/chongshengwang/ads/short_term_prediction/outputs/ablation_no_fhv_2024'
os.makedirs(OUT_DIR, exist_ok=True)

CAT_FEATURES     = ['cab_type']
ZONE_FEATURES    = ['PULocationID']
TIME_FEATURES    = ['hour', 'day_of_week', 'month', 'is_weekend', 'is_rush_hour', 'is_public_holiday']
WEATHER_FEATURES = ['precipitation', 'temperature', 'wind_speed', 'visibility', 'rain_category']
LAG_FEATURES     = ['lag_1', 'lag_2', 'lag_24', 'lag_168']
ALL_FEATURES     = CAT_FEATURES + ZONE_FEATURES + TIME_FEATURES + WEATHER_FEATURES + LAG_FEATURES
TARGET           = 'trip_count'

VAL_MONTHS  = [1, 2, 3]
TEST_MONTHS = [4, 5, 6, 7, 8, 9, 10, 11, 12]

VARIANTS = {
    'A_full':              ALL_FEATURES,
    'B_no_weather':        CAT_FEATURES + ZONE_FEATURES + TIME_FEATURES + LAG_FEATURES,
    'D_no_lag_weather':    CAT_FEATURES + ZONE_FEATURES + TIME_FEATURES + WEATHER_FEATURES,
    'E_no_lag_no_weather': CAT_FEATURES + ZONE_FEATURES + TIME_FEATURES,
}
NO_LAG_VARIANTS = {'D_no_lag_weather', 'E_no_lag_no_weather'}

def make_model():
    return xgb.XGBRegressor(
        n_estimators=1500, learning_rate=0.05, max_depth=8,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        objective='reg:squarederror', tree_method='hist', device='cuda',
        enable_categorical=True, early_stopping_rounds=20,
        eval_metric='mae', random_state=42,
    )



load_cols = list(dict.fromkeys(ALL_FEATURES + [TARGET, 'year', 'month']))
df = pd.read_parquet(DATA_PATH, columns=load_cols).reset_index(drop=True)

# Remove FHV
df = df[df['cab_type'] != 'fhv'].copy()
df['cab_type'] = df['cab_type'].astype('category')



year_arr  = df['year'].to_numpy()
month_arr = df['month'].to_numpy()
lag_arr   = df['lag_168'].to_numpy()
lag_ok    = ~np.isnan(lag_arr.astype(float))

train_lag = np.isin(year_arr, [2022, 2023]) & lag_ok
val_lag   = (year_arr == 2024) & np.isin(month_arr, VAL_MONTHS) & lag_ok
test_lag  = (year_arr == 2024) & np.isin(month_arr, TEST_MONTHS) & lag_ok
train_nolag = np.isin(year_arr, [2022, 2023])
val_nolag   = (year_arr == 2024) & np.isin(month_arr, VAL_MONTHS)
test_nolag  = (year_arr == 2024) & np.isin(month_arr, TEST_MONTHS)

for variant_key, features in VARIANTS.items():
    model_path = os.path.join(OUT_DIR, f'{variant_key}_model.ubj')
    if os.path.exists(model_path):
        continue

    no_lag = variant_key in NO_LAG_VARIANTS
    tr_mask = train_nolag if no_lag else train_lag
    va_mask = val_nolag   if no_lag else val_lag
    te_mask = test_nolag  if no_lag else test_lag

    X_train, y_train = df.loc[tr_mask, features], df.loc[tr_mask, TARGET]
    X_val,   y_val   = df.loc[va_mask, features], df.loc[va_mask, TARGET]
    X_test,  y_test  = df.loc[te_mask, features], df.loc[te_mask, TARGET]

    model = make_model()
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=200)

    pred = model.predict(X_test).clip(min=0)
    mae  = mean_absolute_error(y_test.values, pred)

    model.save_model(model_path)

    del X_train, y_train, X_val, y_val, X_test, y_test, pred, model
    gc.collect()


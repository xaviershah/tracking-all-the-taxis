import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = '/Users/valentinavelasco/year 4/DATA SCIENCE/coursework/group/randomForest/dataSets/all_types_no_fhv_active_yellow.parquet'
OUT_DIR   = '/Users/valentinavelasco/year 4/DATA SCIENCE/coursework/group/randomForest/outputs - Dataset'
os.makedirs(OUT_DIR, exist_ok=True)

# features 
CAT_FEATURES      = ['cab_type']
ZONE_FEATURES     = ['PULocationID']
TIME_FEATURES     = ['hour', 'day_of_week', 'month', 'is_weekend', 'is_rush_hour']
WEATHER_FEATURES  = ['precipitation', 'temperature', 'wind_speed', 'visibility', 'rain_category']
LAG_FEATURES      = ['lag_1', 'lag_2', 'lag_24', 'lag_168']

FEATURES = CAT_FEATURES + ZONE_FEATURES + TIME_FEATURES + WEATHER_FEATURES + LAG_FEATURES
TARGET   = 'trip_count'

print("Loading dataset...")
df = pd.read_parquet(DATA_PATH, columns=FEATURES + [TARGET, 'year'])
print(f"  Loaded: {len(df):,} rows")

# drop rows missing lag_168 (first 168h of each zone/type — no history yet)
df = df.dropna(subset=['lag_168'])
print(f"  After dropping lag NaN rows: {len(df):,} rows")

# split
train = df[df['year'].isin([2022, 2023])].copy()

val = df[
    (df['year'] == 2024) &
    (df['month'] <= 6)
].copy()

test = df[
    (df['year'] == 2024) &
    (df['month'] > 6)
].copy()

print(f"\n  Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")

X_train_raw, y_train = train[FEATURES], train[TARGET]
X_val_raw,   y_val   = val[FEATURES],   val[TARGET]
X_test_raw,  y_test  = test[FEATURES],  test[TARGET]

# Keep copies for outputs / breakdowns
val_meta  = val.copy()
test_meta = test.copy()

del df, train  # free memory before training

# encode categorical features

print("\nEncoding categorical features...")

# random Forest in sklearn cannot consume pandas categorical directly,
# so one-hot encode cab_type and align columns across splits.
X_train = pd.get_dummies(X_train_raw, columns=['cab_type'], dtype='int8')
X_val   = pd.get_dummies(X_val_raw,   columns=['cab_type'], dtype='int8')
X_test  = pd.get_dummies(X_test_raw,  columns=['cab_type'], dtype='int8')

X_val  = X_val.reindex(columns=X_train.columns, fill_value=0)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

print(f"  Encoded train shape: {X_train.shape}")
print(f"  Encoded val shape:   {X_val.shape}")
print(f"  Encoded test shape:  {X_test.shape}")

del X_train_raw, X_val_raw, X_test_raw

#metrics

def mape(actual, predicted):
    actual    = np.asarray(actual)
    predicted = np.asarray(predicted)
    mask = actual > 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def compute_metrics(actual, predicted, label):
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mp   = mape(actual, predicted)
    print(f"  {label}: MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mp:.2f}%")
    return {
        'split': label,
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'MAPE': round(mp, 4)
    }

# baseline: naive lag_1

print("\nBaseline (lag_1 as predictor):")
results = [
    compute_metrics(y_val.values,  val_meta['lag_1'].clip(lower=0).values,  'baseline val'),
    compute_metrics(y_test.values, test_meta['lag_1'].clip(lower=0).values, 'baseline test'),
]

#train Random Forest 
print("\nTraining Random Forest...")
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_leaf=10,
    max_features=0.7,
    n_jobs=-1,
    random_state=42,
    verbose=2
)

model.fit(X_train, y_train)

joblib.dump(model, os.path.join(OUT_DIR, 'random_forest_model.joblib'))
print("  Model saved.")


#evaluate pred
val_pred  = model.predict(X_val).clip(min=0)
test_pred = model.predict(X_test).clip(min=0)

print("\nRandom Forest metrics:")
results += [
    compute_metrics(y_val.values,  val_pred,  'rf val  (2024)'),
    compute_metrics(y_test.values, test_pred, 'rf test (2025)'),
]

#per cab type breakdown on the test set
print("\nTest set breakdown by cab_type:")
test_copy = test_meta.copy()
test_copy['predicted'] = test_pred

for cab in sorted(test_copy['cab_type'].dropna().unique()):
    mask = test_copy['cab_type'] == cab
    r = compute_metrics(
        test_copy.loc[mask, TARGET].values,
        test_copy.loc[mask, 'predicted'].values,
        f'  {cab} test'
    )
    results.append(r)

pd.DataFrame(results).to_csv(
    os.path.join(OUT_DIR, 'random_forest_metrics.csv'),
    index=False
)

pred_out = pd.concat([
    val_meta.assign(predicted=val_pred,   split='val'),
    test_meta.assign(predicted=test_pred, split='test'),
])[['cab_type', 'PULocationID', 'year', TARGET, 'predicted', 'split']]

pred_out = pred_out.rename(columns={TARGET: 'actual'})
pred_out.to_parquet(
    os.path.join(OUT_DIR, 'random_forest_predictions.parquet'),
    index=False
)

# feature importance to assess results of random forest, is weather significant?

importance = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 8))
importance.tail(25).plot.barh(ax=ax, color='#4A90D9', edgecolor='none')
ax.set_title('Random Forest Feature Importance (Top 25)')
ax.set_xlabel('Importance score')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'random_forest_feature_importance.png'), dpi=150)
plt.close()

importance.sort_values(ascending=False).to_csv(
    os.path.join(OUT_DIR, "random_forest_feature_importance.csv")
)

print(f"\nDone. All outputs saved to {OUT_DIR}/")
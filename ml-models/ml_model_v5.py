import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('data/tickets_dataset_datetime_converted.csv')

# Parse datetime and drop missing
df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
df = df.dropna(subset=['start_date'])

# Aggregate daily ticket volume
daily_df = df.groupby(df['start_date'].dt.date).size().reset_index(name='ticket_volume_day')
daily_df['start_date'] = pd.to_datetime(daily_df['start_date'])

# Extract holiday and major event flags per day
holiday_flags = df.groupby(df['start_date'].dt.date)['is_holiday'].max().reset_index(name='is_holiday')
major_event_flags = df.groupby(df['start_date'].dt.date)['major_event'].max().reset_index(name='major_event')

holiday_flags['start_date'] = pd.to_datetime(holiday_flags['start_date'])
major_event_flags['start_date'] = pd.to_datetime(major_event_flags['start_date'])

# Merge holiday and major event flags into daily_df
daily_df = daily_df.merge(holiday_flags, on='start_date', how='left')
daily_df = daily_df.merge(major_event_flags, on='start_date', how='left')

# Fill any missing values with 0 (no holiday or event)
daily_df['is_holiday'] = daily_df['is_holiday'].fillna(0).astype(int)
daily_df['major_event'] = daily_df['major_event'].fillna(0).astype(int)

# Feature Engineering
daily_df['prev_day_volume'] = daily_df['ticket_volume_day'].shift(1)
daily_df['prev_2day_volume'] = daily_df['ticket_volume_day'].shift(2)
daily_df['prev_3day_volume'] = daily_df['ticket_volume_day'].shift(3)
daily_df['prev_7day_volume'] = daily_df['ticket_volume_day'].shift(7)

daily_df['rolling_avg_7d'] = daily_df['ticket_volume_day'].rolling(window=7).mean()
daily_df['rolling_avg_14d'] = daily_df['ticket_volume_day'].rolling(window=14).mean()
daily_df['rolling_avg_30d'] = daily_df['ticket_volume_day'].rolling(window=30).mean()

daily_df['ewm_7d'] = daily_df['ticket_volume_day'].ewm(span=7).mean()
daily_df['ewm_14d'] = daily_df['ticket_volume_day'].ewm(span=14).mean()

# Date-based features
daily_df['day_of_week'] = daily_df['start_date'].dt.dayofweek
daily_df['month'] = daily_df['start_date'].dt.month
daily_df['is_weekend'] = daily_df['day_of_week'].isin([5, 6]).astype(int)

# Drop NA rows due to shifts and rolling calculations
daily_df = daily_df.dropna()

# Define features and target including new holiday and major event flags
features = [
    'prev_day_volume', 'prev_2day_volume', 'prev_3day_volume', 'prev_7day_volume',
    'rolling_avg_7d', 'rolling_avg_14d', 'rolling_avg_30d',
    'ewm_7d', 'ewm_14d',
    'day_of_week', 'month', 'is_weekend',
    'is_holiday', 'major_event'
]

X = daily_df[features]
y = daily_df['ticket_volume_day']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Time series split (5 folds)
tscv = TimeSeriesSplit(n_splits=5)
train_idx, test_idx = list(tscv.split(X_scaled))[-1]
X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# Initialize XGBRegressor with early stopping
model = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

# Train with early stopping
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=20,
    verbose=False
)

# Predict and evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("📊 Model Evaluation Results (Improved with Holiday & Event Flags)")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.2f}")
print(f"Accuracy (Model Score): {model.score(X_test, y_test):.2f}")

# Feature importance plot
importance = model.feature_importances_
plt.barh(features, importance)
plt.xlabel('Feature Importance')
plt.title('XGBoost Feature Importance')
plt.show()

# Predict next day (use last row features scaled)
last_row = X.iloc[-1:].values
last_row_scaled = scaler.transform(last_row)
next_day_pred = model.predict(last_row_scaled)[0]
print(f"\n📅 Predicted Ticket Volume for Next Day: {next_day_pred:.0f}")

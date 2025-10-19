import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('data/tickets_dataset_datetime_converted.csv')

# Convert date columns to datetime
df['start_date'] = pd.to_datetime(df['start_date'])
df['resolve_date'] = pd.to_datetime(df['resolve_date'])

# Fill missing values
df['major_event'] = df['major_event'].fillna('None')
df['customer_impact'] = df['customer_impact'].fillna('Unknown')

# Encode categorical variables
cat_cols = ['priority', 'category', 'sub_category', 'department', 'assignee_team',
            'sla_status', 'customer_impact', 'day_of_week']

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

df['major_event'] = LabelEncoder().fit_transform(df['major_event'].astype(str))

# Date-derived features
df['week_of_year'] = df['start_date'].dt.isocalendar().week
df['quarter'] = df['start_date'].dt.quarter
df['season'] = df['start_date'].dt.month % 12 // 3 + 1

# Cyclical encoding
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['day_sin'] = np.sin(2 * np.pi * df['start_date'].dt.day / 31)
df['day_cos'] = np.cos(2 * np.pi * df['start_date'].dt.day / 31)

# Sort and reset index
df = df.sort_values('start_date').reset_index(drop=True)

# Lag and rolling features
df['ticket_volume_day_lag1'] = df['ticket_volume_day'].shift(1).fillna(0)
df['ticket_volume_day_lag7'] = df['ticket_volume_day'].shift(7).fillna(0)
df['ticket_volume_rolling3'] = df['ticket_volume_day'].rolling(window=3).mean().fillna(0)
df['ticket_volume_rolling7'] = df['ticket_volume_day'].rolling(window=7).mean().fillna(0)
df['ticket_volume_sum3'] = df['ticket_volume_day'].rolling(window=3).sum().fillna(0)
df['ticket_volume_sum7'] = df['ticket_volume_day'].rolling(window=7).sum().fillna(0)

# Interaction features
df['priority_category'] = LabelEncoder().fit_transform(
    df['priority'].astype(str) + "_" + df['category'].astype(str))
df['team_department'] = LabelEncoder().fit_transform(
    df['assignee_team'].astype(str) + "_" + df['department'].astype(str))

# Final feature list
features = ['priority', 'category', 'sub_category', 'department', 'assignee_team',
            'sla_status', 'resolution_time_hrs', 'customer_impact', 'major_event',
            'is_holiday', 'is_weekend', 'day_of_week', 'month_sin', 'month_cos',
            'day_sin', 'day_cos', 'week_of_year', 'quarter', 'season',
            'ticket_volume_day_lag1', 'ticket_volume_day_lag7',
            'ticket_volume_rolling3', 'ticket_volume_rolling7',
            'ticket_volume_sum3', 'ticket_volume_sum7',
            'priority_category', 'team_department']

X = df[features]
y = df['ticket_volume_day']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base regressors
base_estimators = [
    ('xgb', xgb.XGBRegressor(n_estimators=100, objective='reg:squarederror', random_state=42)),
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('lgb', lgb.LGBMRegressor(n_estimators=100, random_state=42))
]

# Define stacking ensemble
stacking_regressor = StackingRegressor(
    estimators=base_estimators,
    final_estimator=LinearRegression()
)

# Train model
stacking_regressor.fit(X_train, y_train)

# Predict and evaluate
y_pred = stacking_regressor.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # compatible across sklearn versions
r2 = r2_score(y_test, y_pred)

print(f"Stacking Ensemble RMSE: {rmse:.2f}")
print(f"Stacking Ensemble R2: {r2:.2f}")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor, plot_importance
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('data/tickets_dataset_datetime_converted.csv')
df['start_date'] = pd.to_datetime(df['start_date'])

# Fill missing values
df['major_event'] = df['major_event'].fillna('None')
df['customer_impact'] = df['customer_impact'].fillna('Unknown')

# Encode categorical features
cat_cols = ['priority', 'category', 'sub_category', 'department', 'assignee_team',
            'sla_status', 'customer_impact', 'day_of_week']
df['day_of_week'] = df['day_of_week'].astype(str)
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col])
df['major_event'] = LabelEncoder().fit_transform(df['major_event'].astype(str))

# Date-derived features
df['week_of_year'] = df['start_date'].dt.isocalendar().week
df['quarter'] = df['start_date'].dt.quarter
df['season'] = df['start_date'].dt.month % 12 // 3 + 1  # 1=Winter, 2=Spring, etc.

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
y = np.log1p(df['ticket_volume_day'])  # log-transform target

# Time-based cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# Grid search for hyperparameter tuning
param_grid = {
    'n_estimators': [300, 500],
    'max_depth': [8, 10],
    'learning_rate': [0.03, 0.05],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9]
}

grid_search = GridSearchCV(
    estimator=XGBRegressor(objective='reg:squarederror', random_state=42),
    param_grid=param_grid,
    cv=tscv,
    scoring='neg_mean_squared_error',
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X, y)
best_model = grid_search.best_estimator_

# Train-test split for final evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
best_model.fit(X_train, y_train)
y_pred_log = best_model.predict(X_test)

# Inverse transform predictions
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)
print(f'Final RMSE: {rmse:.2f}')
print(f'Final R-squared (R2): {r2:.2f}')

# Feature importance plot
plot_importance(best_model, max_num_features=15)
plt.title("Top 15 Feature Importances")
plt.tight_layout()
plt.show()

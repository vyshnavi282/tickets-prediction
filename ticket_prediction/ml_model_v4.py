# -------------------------------
# Ticket Volume Prediction Project
# -------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv('data/tickets_dataset_datetime_converted.csv')

# Convert datetime column
df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')

# Drop rows with missing dates
df = df.dropna(subset=['start_date'])

# -------------------------------
# 2. Aggregate Ticket Volume by Date
# -------------------------------
# Count number of tickets created per day
daily_df = df.groupby(df['start_date'].dt.date).size().reset_index(name='ticket_volume_day')
daily_df['start_date'] = pd.to_datetime(daily_df['start_date'])

# -------------------------------
# 3. Feature Engineering
# -------------------------------
# Lag features (previous days' ticket volumes)
daily_df['prev_day_volume'] = daily_df['ticket_volume_day'].shift(1)
daily_df['prev_7day_volume'] = daily_df['ticket_volume_day'].shift(7)

# Rolling averages for weekly & monthly trends
daily_df['rolling_avg_7d'] = daily_df['ticket_volume_day'].rolling(window=7).mean()
daily_df['rolling_avg_30d'] = daily_df['ticket_volume_day'].rolling(window=30).mean()

# Date-based features
daily_df['day_of_week'] = daily_df['start_date'].dt.dayofweek
daily_df['month'] = daily_df['start_date'].dt.month
daily_df['is_weekend'] = daily_df['day_of_week'].isin([5, 6]).astype(int)

# -------------------------------
# 4. Handle Missing Values (from shifts and rolling)
# -------------------------------
daily_df = daily_df.dropna()

# -------------------------------
# 5. Define Features and Target
# -------------------------------
features = [
    'prev_day_volume', 'prev_7day_volume', 'rolling_avg_7d', 'rolling_avg_30d',
    'day_of_week', 'month', 'is_weekend'
]
X = daily_df[features]
y = daily_df['ticket_volume_day']

# -------------------------------
# 6. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# -------------------------------
# 7. Initialize and Train Model
# -------------------------------
model = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)
model.fit(X_train, y_train)

# -------------------------------
# 8. Predict and Evaluate
# -------------------------------
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("📊 Model Evaluation Results")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# Optional accuracy-style metric for reference
acc = model.score(X_test, y_test)
print(f"Accuracy (Model Score): {acc:.2f}")

# -------------------------------
# 9. Predict Future Ticket Volume (Next Day)
# -------------------------------
last_row = daily_df.iloc[-1:]
next_day_features = last_row[features]
next_day_pred = model.predict(next_day_features)[0]

print(f"\n📅 Predicted Ticket Volume for Next Day: {next_day_pred:.0f}")


'''import pandas as pd

df = pd.read_csv('tickets_dataset_datetime_converted.csv')
print(df.columns.tolist())'''

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('data/tickets_dataset_datetime_converted.csv')

# Parse datetime and drop missing
df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
df = df.dropna(subset=['start_date'])

# Convert major_event column to binary flag 0/1
df['major_event'] = df['major_event'].fillna('')
df['major_event'] = df['major_event'].apply(lambda x: 1 if str(x).strip() != '' else 0)

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

# Fill missing values
daily_df['is_holiday'] = daily_df['is_holiday'].fillna(0).astype(int)
daily_df['major_event'] = daily_df['major_event'].fillna(0).astype(int)

# Feature Engineering function (for historic data)
def create_features(df):
    df['prev_day_volume'] = df['ticket_volume_day'].shift(1)
    df['prev_2day_volume'] = df['ticket_volume_day'].shift(2)
    df['prev_3day_volume'] = df['ticket_volume_day'].shift(3)
    df['prev_7day_volume'] = df['ticket_volume_day'].shift(7)

    df['rolling_avg_7d'] = df['ticket_volume_day'].rolling(window=7).mean()
    df['rolling_avg_14d'] = df['ticket_volume_day'].rolling(window=14).mean()
    df['rolling_avg_30d'] = df['ticket_volume_day'].rolling(window=30).mean()

    df['ewm_7d'] = df['ticket_volume_day'].ewm(span=7).mean()
    df['ewm_14d'] = df['ticket_volume_day'].ewm(span=14).mean()

    df['day_of_week'] = df['start_date'].dt.dayofweek
    df['month'] = df['start_date'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
    return df

daily_df = create_features(daily_df)
daily_df = daily_df.dropna()

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

# Train XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'eta': 0.05,
    'seed': 42
}

bst = xgb.train(
    params,
    dtrain,
    num_boost_round=500,
    evals=[(dtest,'eval')],
    early_stopping_rounds=20,
    verbose_eval=False
)

# Multi-day prediction function
def predict_range(bst, scaler, daily_df, start_date, end_date, holiday_flags, major_event_flags):
    # Copy current data for feature construction
    data = daily_df.copy()
    predictions = []

    dates = pd.date_range(start_date, end_date)

    for date in dates:
        # Prepare feature row for the date
        prev_day = data[data['start_date'] == (date - pd.Timedelta(days=1))]
        prev_2day = data[data['start_date'] == (date - pd.Timedelta(days=2))]
        prev_3day = data[data['start_date'] == (date - pd.Timedelta(days=3))]
        prev_7day = data[data['start_date'] == (date - pd.Timedelta(days=7))]

        # Calculate rolling and ewm averages from data
        past_window = data[(data['start_date'] < date) & (data['start_date'] >= date - pd.Timedelta(days=30))]
        rolling_avg_7d = past_window.tail(7)['ticket_volume_day'].mean() if len(past_window) >= 7 else np.nan
        rolling_avg_14d = past_window.tail(14)['ticket_volume_day'].mean() if len(past_window) >= 14 else np.nan
        rolling_avg_30d = past_window['ticket_volume_day'].mean() if len(past_window) >= 1 else np.nan

        ewm_7d = past_window['ticket_volume_day'].ewm(span=7).mean().iloc[-1] if len(past_window) >= 1 else np.nan
        ewm_14d = past_window['ticket_volume_day'].ewm(span=14).mean().iloc[-1] if len(past_window) >= 1 else np.nan

        # Day features
        day_of_week = date.dayofweek
        month = date.month
        is_weekend = int(day_of_week in [5,6])

        # Holiday & major event flags
        is_holiday = holiday_flags[holiday_flags['start_date'] == date]['is_holiday']
        is_holiday = int(is_holiday.values[0]) if not is_holiday.empty else 0

        major_event = major_event_flags[major_event_flags['start_date'] == date]['major_event']
        major_event = int(major_event.values[0]) if not major_event.empty else 0

        # Construct feature vector (handle missing lag values by using last available or zeros)
        feature_vector = [
            prev_day['ticket_volume_day'].values[0] if not prev_day.empty else 0,
            prev_2day['ticket_volume_day'].values[0] if not prev_2day.empty else 0,
            prev_3day['ticket_volume_day'].values[0] if not prev_3day.empty else 0,
            prev_7day['ticket_volume_day'].values[0] if not prev_7day.empty else 0,
            rolling_avg_7d if not pd.isna(rolling_avg_7d) else 0,
            rolling_avg_14d if not pd.isna(rolling_avg_14d) else 0,
            rolling_avg_30d if not pd.isna(rolling_avg_30d) else 0,
            ewm_7d if not pd.isna(ewm_7d) else 0,
            ewm_14d if not pd.isna(ewm_14d) else 0,
            day_of_week,
            month,
            is_weekend,
            is_holiday,
            major_event
        ]

        # Scale features and predict
        scaled_features = scaler.transform([feature_vector])
        dmatrix = xgb.DMatrix(scaled_features)
        pred = bst.predict(dmatrix)[0]

        # Append prediction
        predictions.append({'date': date, 'predicted_ticket_volume': pred})

        # Append to data for iterative predictions
        new_row = pd.DataFrame({
            'start_date': [date],
            'ticket_volume_day': [pred],
            'is_holiday': [is_holiday],
            'major_event': [major_event]
        })

        data = pd.concat([data, new_row], ignore_index=True)
        data = create_features(data)
        data = data.dropna()

    return pd.DataFrame(predictions)

# Example usage: predict for next 7 days from last data day
last_date = daily_df['start_date'].max()
predictions_df = predict_range(bst, scaler, daily_df, last_date + pd.Timedelta(days=1), last_date + pd.Timedelta(days=7), holiday_flags, major_event_flags)

print(predictions_df)

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

class TicketVolumePredictor:
    def __init__(self, csv_path):
        # Load the dataset from the specified CSV file path
        self.df = pd.read_csv(csv_path)
        # Placeholder for the daily aggregated ticket volume dataframe
        self.daily_df = None
        # Placeholders for holiday and major event flags aggregated by day
        self.holiday_flags = None
        self.major_event_flags = None
        # Initialize a scaler to normalize feature values
        self.scaler = StandardScaler()
        # Placeholder for the trained XGBoost model
        self.bst = None
        # List of feature names used as input to the model
        self.features = [
            'prev_day_volume', 'prev_2day_volume', 'prev_3day_volume', 'prev_7day_volume',
            'rolling_avg_7d', 'rolling_avg_14d', 'rolling_avg_30d',
            'ewm_7d', 'ewm_14d',
            'day_of_week', 'month', 'is_weekend',
            'is_holiday', 'major_event'
        ]
        # Prepare the dataset and train the model upon initialization
        self._prepare_data()
        self._train_model()

    def _prepare_data(self):
        # Work on a copy of the original dataset to avoid modification
        df = self.df.copy()
        # Convert 'start_date' strings to datetime objects, invalid dates become NaT
        df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
        # Drop rows where 'start_date' conversion failed (NaT)
        df = df.dropna(subset=['start_date'])

        # Convert 'major_event' column to binary flags (1 if event name is present, else 0)
        df['major_event'] = df['major_event'].fillna('').apply(lambda x: 1 if str(x).strip() != '' else 0)

        # Aggregate the number of tickets by day, creating 'ticket_volume_day' column
        daily_df = df.groupby(df['start_date'].dt.date).size().reset_index(name='ticket_volume_day')
        # Convert grouped dates back to datetime format for consistency
        daily_df['start_date'] = pd.to_datetime(daily_df['start_date'])

        # Aggregate holiday flags per day using max to indicate holiday presence
        self.holiday_flags = df.groupby(df['start_date'].dt.date)
        ['is_holiday'].max().reset_index(name='is_holiday')
        # Aggregate major event flags per day similarly
        self.major_event_flags = df.groupby(df['start_date'].dt.date)
        ['major_event'].max().reset_index(name='major_event')

        # Convert dates in flags back to datetime
        self.holiday_flags['start_date'] = pd.to_datetime(self.holiday_flags['start_date'])
        self.major_event_flags['start_date'] = pd.to_datetime(self.major_event_flags['start_date'])

        # Merge holiday flags into the daily aggregated ticket volume dataframe
        daily_df = daily_df.merge(self.holiday_flags, on='start_date', how='left')
        # Merge major event flags similarly
        daily_df = daily_df.merge(self.major_event_flags, on='start_date', how='left')

        # Replace missing flag values with 0 and convert them to integers
        daily_df['is_holiday'] = daily_df['is_holiday'].fillna(0).astype(int)
        daily_df['major_event'] = daily_df['major_event'].fillna(0).astype(int)

        # Create lag, rolling average, and other time series features for modeling
        daily_df = self._create_features(daily_df)
        # Drop rows with any NaN caused by shifting or rolling computations
        self.daily_df = daily_df.dropna()

    def _create_features(self, df):
        # Work on a copy of the dataframe to avoid side effects
        df = df.copy()
        # Lag features for ticket volume on previous 1, 2, 3, and 7 days
        df['prev_day_volume'] = df['ticket_volume_day'].shift(1)
        df['prev_2day_volume'] = df['ticket_volume_day'].shift(2)
        df['prev_3day_volume'] = df['ticket_volume_day'].shift(3)
        df['prev_7day_volume'] = df['ticket_volume_day'].shift(7)

        # Rolling average ticket volume over 7, 14, and 30 day windows
        df['rolling_avg_7d'] = df['ticket_volume_day'].rolling(window=7).mean()
        df['rolling_avg_14d'] = df['ticket_volume_day'].rolling(window=14).mean()
        df['rolling_avg_30d'] = df['ticket_volume_day'].rolling(window=30).mean()

        # Exponentially weighted moving averages (EWMA) with span of 7 and 14 days
        df['ewm_7d'] = df['ticket_volume_day'].ewm(span=7).mean()
        df['ewm_14d'] = df['ticket_volume_day'].ewm(span=14).mean()

        # Extract day of week (Monday=0), month number, and weekend flag (Saturday=5, Sunday=6)
        df['day_of_week'] = df['start_date'].dt.dayofweek
        df['month'] = df['start_date'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        return df

    def _train_model(self):
        # Select feature columns and target variable for model training
        X = self.daily_df[self.features]
        y = self.daily_df['ticket_volume_day']

        # Scale features to zero mean and unit variance for model input
        X_scaled = self.scaler.fit_transform(X)

        # Use TimeSeriesSplit for time-aware cross-validation on the scaled features
        tscv = TimeSeriesSplit(n_splits=5)
        # Take the last split to define train and test dataset indices
        train_idx, test_idx = list(tscv.split(X_scaled))[-1]
        # Split the scaled features and labels accordingly
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Create XGBoost data matrices from train and test data
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        # Define XGBoost regression parameters including learning rate, tree depth, and seed
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'eta': 0.05,
            'seed': 42
        }

        # Train the XGBoost model with early stopping on validation (test) set, max 500 rounds
        self.bst = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=[(dtest, 'eval')],
            early_stopping_rounds=20,
            verbose_eval=False
        )

        # Predict on test data, compute and print RMSE and R-squared evaluation metrics
        y_pred = self.bst.predict(dtest)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f"Model Evaluation:\nRMSE: {rmse:.2f}\nR-squared: {r2:.2f}")

    def predict_range(self, start_date, end_date):
        # Copy the prepared daily dataset to work on for prediction
        data = self.daily_df.copy()
        predictions = []
        # Create a range of dates from start_date to end_date inclusive
        dates = pd.date_range(start_date, end_date)

        # Iterate over each date to predict ticket volume
        for date in dates:
            # Retrieve previous day volumes for lag features, handle missing dates by empty results
            prev_day = data[data['start_date'] == (date - pd.Timedelta(days=1))]
            prev_2day = data[data['start_date'] == (date - pd.Timedelta(days=2))]
            prev_3day = data[data['start_date'] == (date - pd.Timedelta(days=3))]
            prev_7day = data[data['start_date'] == (date - pd.Timedelta(days=7))]

            # Define a rolling window of the past 30 days data up to prediction date
            past_window = data[(data['start_date'] < date) & (data['start_date'] >= date - pd.Timedelta(days=30))]
            # Calculate rolling averages for 7, 14, and 30 days if enough data exists, else zero
            rolling_avg_7d = past_window.tail(7)
            ['ticket_volume_day'].mean() if len(past_window) >= 7 else 0
            rolling_avg_14d = past_window.tail(14)
            ['ticket_volume_day'].mean() if len(past_window) >= 14 else 0
            rolling_avg_30d = past_window['ticket_volume_day'].mean() if len(past_window) >= 1 else 0

            # Calculate exponentially weighted moving averages for 7 and 14 days spans if data exists, else zero
            ewm_7d = past_window['ticket_volume_day'].ewm(span=7).mean().iloc[-1] if len(past_window) >= 1 else 0
            ewm_14d = past_window['ticket_volume_day'].ewm(span=14).mean().iloc[-1] if len(past_window) >= 1 else 0

            # Extract date features for current date prediction
            day_of_week = date.dayofweek
            month = date.month
            is_weekend = int(day_of_week in [5, 6])

            # Retrieve holiday flag for the current date if present, else 0
            is_holiday = self.holiday_flags[self.holiday_flags['start_date'] == date]['is_holiday']
            is_holiday = int(is_holiday.values[0]) if not is_holiday.empty else 0

            # Retrieve major event flag for the current date if present, else 0
            major_event = self.major_event_flags[self.major_event_flags['start_date'] == date]
            ['major_event']
            major_event = int(major_event.values[0]) if not major_event.empty else 0

            # Construct the feature vector for the current date prediction
            feature_vector = [
                prev_day['ticket_volume_day'].values[0] if not prev_day.empty else 0,
                prev_2day['ticket_volume_day'].values[0] if not prev_2day.empty else 0,
                prev_3day['ticket_volume_day'].values[0] if not prev_3day.empty else 0,
                prev_7day['ticket_volume_day'].values[0] if not prev_7day.empty else 0,
                rolling_avg_7d, rolling_avg_14d, rolling_avg_30d,
                ewm_7d, ewm_14d,
                day_of_week, month, is_weekend,
                is_holiday, major_event
            ]

            # Scale the feature vector before prediction using the trained scaler
            scaled_features = self.scaler.transform([feature_vector])
            # Convert features to DMatrix format for XGBoost prediction
            dmatrix = xgb.DMatrix(scaled_features)
            # Predict ticket volume for the current date
            pred = self.bst.predict(dmatrix)[0]

            # Append the prediction results with formatted date
            predictions.append({'date': date.strftime('%Y-%m-%d'), 'predicted_ticket_volume': pred})

            # Prepare a new row with predicted volume and flags to append to data for next iterations
            new_row = pd.DataFrame({
                'start_date': [date],
                'ticket_volume_day': [pred],
                'is_holiday': [is_holiday],
                'major_event': [major_event]
            })

            # Concatenate the new prediction row to the existing data
            data = pd.concat([data, new_row], ignore_index=True)
            # Recalculate features with the added predicted data row
            data = self._create_features(data)
            # Drop rows with NaN values caused by recalculated lag/rolling features
            data = data.dropna()

        # Return a DataFrame of all predictions in the date range
        return pd.DataFrame(predictions)

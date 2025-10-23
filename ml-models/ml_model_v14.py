import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

class TicketVolumePredictor:
    def __init__(self, csv_path):
        # Load the dataset from CSV
        self.df = pd.read_csv(csv_path)
        self.daily_df = None
        self.holiday_flags = None
        self.major_event_flags = None
        self.scaler = StandardScaler()  # Scaler for feature normalization
        self.bst = None                 # Model placeholder
        # List of feature column names for model input
        self.features = [
            'prev_day_volume', 'prev_2day_volume', 'prev_3day_volume', 'prev_7day_volume',
            'rolling_avg_7d', 'rolling_avg_14d', 'rolling_avg_30d',
            'ewm_7d', 'ewm_14d',
            'day_of_week', 'month', 'is_weekend',
            'is_holiday', 'major_event'
        ]
        # Prepare data and train model on initialization
        self._prepare_data()
        self._train_model()

    def _prepare_data(self):
        # Copy original dataframe to avoid changes
        df = self.df.copy()
        # Convert start_date to datetime and drop invalid dates
        df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
        df = df.dropna(subset=['start_date'])

        # Convert major_event flag to binary 1/0
        df['major_event'] = df['major_event'].fillna('').apply
        (lambda x: 1 if str(x).strip() != '' else 0)

        # Aggregate data daily to count tickets per day
        daily_df = df.groupby(df['start_date'].dt.date).size().reset_index(name='ticket_volume_day')
        daily_df['start_date'] = pd.to_datetime(daily_df['start_date'])

        # Get holiday flags per day (max in case multiple entries)
        self.holiday_flags = df.groupby(df['start_date'].dt.date)
        ['is_holiday'].max().reset_index(name='is_holiday')
        self.major_event_flags = df.groupby(df['start_date'].dt.date)
        ['major_event'].max().reset_index(name='major_event')

        self.holiday_flags['start_date'] = pd.to_datetime(self.holiday_flags['start_date'])
        self.major_event_flags['start_date'] = pd.to_datetime(self.major_event_flags['start_date'])

        # Merge holiday and event flags to daily dataset
        daily_df = daily_df.merge(self.holiday_flags, on='start_date', how='left')
        daily_df = daily_df.merge(self.major_event_flags, on='start_date', how='left')

        daily_df['is_holiday'] = daily_df['is_holiday'].fillna(0).astype(int)
        daily_df['major_event'] = daily_df['major_event'].fillna(0).astype(int)

        # Create engineered features such as lag volumes, rolling averages, etc.
        daily_df = self._create_features(daily_df)
        # Drop rows with NaN values created by shifting or rolling
        self.daily_df = daily_df.dropna()

    def _create_features(self, df):
        df = df.copy()
        # Lag features for previous ticket volumes on specific days
        df['prev_day_volume'] = df['ticket_volume_day'].shift(1)
        df['prev_2day_volume'] = df['ticket_volume_day'].shift(2)
        df['prev_3day_volume'] = df['ticket_volume_day'].shift(3)
        df['prev_7day_volume'] = df['ticket_volume_day'].shift(7)

        # Rolling mean features for weekly, biweekly, and monthly periods
        df['rolling_avg_7d'] = df['ticket_volume_day'].rolling(window=7).mean()
        df['rolling_avg_14d'] = df['ticket_volume_day'].rolling(window=14).mean()
        df['rolling_avg_30d'] = df['ticket_volume_day'].rolling(window=30).mean()

        # Exponentially weighted moving averages for two spans
        df['ewm_7d'] = df['ticket_volume_day'].ewm(span=7).mean()
        df['ewm_14d'] = df['ticket_volume_day'].ewm(span=14).mean()

        # Date-related features
        df['day_of_week'] = df['start_date'].dt.dayofweek  # Monday=0, Sunday=6
        df['month'] = df['start_date'].dt.month            # Month number 1-12
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)  # Weekend flag

        return df

    def _train_model(self):
        # Select features and target from prepared dataframe
        X = self.daily_df[self.features]
        y = self.daily_df['ticket_volume_day']

        # Scale features to zero mean and unit variance
        X_scaled = self.scaler.fit_transform(X)

        # TimeSeriesSplit cross-validation for time series data
        tscv = TimeSeriesSplit(n_splits=5)
        # Use the last split for training/testing model
        train_idx, test_idx = list(tscv.split(X_scaled))[-1]
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Prepare data matrices for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        # XGBoost training parameters
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'eta': 0.05,
            'seed': 42
        }

        # Train model with early stopping on eval set
        self.bst = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=[(dtest, 'eval')],
            early_stopping_rounds=20,
            verbose_eval=False
        )

        # Predict and evaluate on the test set
        y_pred = self.bst.predict(dtest)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f"Model Evaluation:\nRMSE: {rmse:.2f}\nR-squared: {r2:.2f}")

    def predict_range(self, start_date, end_date):
        # Copy existing prepared daily data
        data = self.daily_df.copy()
        predictions = []
        dates = pd.date_range(start_date, end_date)

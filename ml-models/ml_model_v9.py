import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb


class TicketVolumePredictor:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.daily_df = None
        self.holiday_flags = None
        self.major_event_flags = None
        self.scaler = StandardScaler()
        self.bst = None
        self.features = [
            'prev_day_volume', 'prev_2day_volume', 'prev_3day_volume', 'prev_7day_volume',
            'rolling_avg_7d', 'rolling_avg_14d', 'rolling_avg_30d',
            'ewm_7d', 'ewm_14d',
            'day_of_week', 'month', 'is_weekend',
            'is_holiday', 'major_event'
        ]
        self._prepare_data()
        self._train_model()


    def _prepare_data(self):
        df = self.df.copy()
        df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
        df = df.dropna(subset=['start_date'])

        df['major_event'] = df['major_event'].fillna('').apply(lambda x: 1 if str(x).strip() != '' else 0)

        daily_df = df.groupby(df['start_date'].dt.date).size().reset_index(name='ticket_volume_day')
        daily_df['start_date'] = pd.to_datetime(daily_df['start_date'])

        self.holiday_flags = df.groupby(df['start_date'].dt.date)['is_holiday'].max().reset_index(name='is_holiday')
        self.major_event_flags = df.groupby(df['start_date'].dt.date)['major_event'].max().reset_index(name='major_event')

        self.holiday_flags['start_date'] = pd.to_datetime(self.holiday_flags['start_date'])
        self.major_event_flags['start_date'] = pd.to_datetime(self.major_event_flags['start_date'])

        daily_df = daily_df.merge(self.holiday_flags, on='start_date', how='left')
        daily_df = daily_df.merge(self.major_event_flags, on='start_date', how='left')

        daily_df['is_holiday'] = daily_df['is_holiday'].fillna(0).astype(int)
        daily_df['major_event'] = daily_df['major_event'].fillna(0).astype(int)

        daily_df = self._create_features(daily_df)
        self.daily_df = daily_df.dropna()


    def _create_features(self, df):
        df = df.copy()
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
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        return df


    def _train_model(self):
        X = self.daily_df[self.features]
        y = self.daily_df['ticket_volume_day']

        X_scaled = self.scaler.fit_transform(X)

        tscv = TimeSeriesSplit(n_splits=5)
        train_idx, test_idx = list(tscv.split(X_scaled))[-1]
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'eta': 0.05,
            'seed': 42
        }

        self.bst = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=[(dtest, 'eval')],
            early_stopping_rounds=20,
            verbose_eval=False
        )

        y_pred = self.bst.predict(dtest)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f"Model Evaluation:\nRMSE: {rmse:.2f}\nR-squared: {r2:.2f}")


    def predict_range(self, start_date, end_date):
        data = self.daily_df.copy()
        predictions = []
        dates = pd.date_range(start_date, end_date)

        for date in dates:
            prev_day = data[data['start_date'] == (date - pd.Timedelta(days=1))]
            prev_2day = data[data['start_date'] == (date - pd.Timedelta(days=2))]
            prev_3day = data[data['start_date'] == (date - pd.Timedelta(days=3))]
            prev_7day = data[data['start_date'] == (date - pd.Timedelta(days=7))]

            past_window = data[(data['start_date'] < date) & (data['start_date'] >= date - pd.Timedelta(days=30))]
            rolling_avg_7d = past_window.tail(7)['ticket_volume_day'].mean() if len(past_window) >= 7 else 0
            rolling_avg_14d = past_window.tail(14)['ticket_volume_day'].mean() if len(past_window) >= 14 else 0
            rolling_avg_30d = past_window['ticket_volume_day'].mean() if len(past_window) >= 1 else 0

            ewm_7d = past_window['ticket_volume_day'].ewm(span=7).mean().iloc[-1] if len(past_window) >= 1 else 0
            ewm_14d = past_window['ticket_volume_day'].ewm(span=14).mean().iloc[-1] if len(past_window) >= 1 else 0

            day_of_week = date.dayofweek
            month = date.month
            is_weekend = int(day_of_week in [5, 6])

            is_holiday = self.holiday_flags[self.holiday_flags['start_date'] == date]['is_holiday']
            is_holiday = int(is_holiday.values[0]) if not is_holiday.empty else 0

            major_event = self.major_event_flags[self.major_event_flags['start_date'] == date]['major_event']
            major_event = int(major_event.values[0]) if not major_event.empty else 0

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

            scaled_features = self.scaler.transform([feature_vector])
            dmatrix = xgb.DMatrix(scaled_features)
            pred = self.bst.predict(dmatrix)[0]

            predictions.append({'date': date.strftime('%Y-%m-%d'), 'predicted_ticket_volume': pred})

            new_row = pd.DataFrame({
                'start_date': [date],
                'ticket_volume_day': [pred],
                'is_holiday': [is_holiday],
                'major_event': [major_event]
            })

            data = pd.concat([data, new_row], ignore_index=True)
            data = self._create_features(data)
            data = data.dropna()

        return pd.DataFrame(predictions)

    def predict_next_n_days(self, days):
        start = pd.Timestamp.today().normalize()  # Current date, no time
        end = start + pd.Timedelta(days=days - 1)  # days including today
        return self.predict_range(start, end)

    def predict_next_2_days(self):
        return self.predict_next_n_days(2)

    def predict_next_30_days(self):
        return self.predict_next_n_days(30)
    
    def predict_next_7_days(self):
        return self.predict_next_n_days(7)

    def predict_this_week(self):
        today = pd.Timestamp.today().normalize()
        # Days remaining in this week (Monday=0 ... Sunday=6)
        days_left = 6 - today.dayofweek + 1  # Including today until Sunday
        return self.predict_next_n_days(days_left)

    def predict_this_month(self):
        today = pd.Timestamp.today().normalize()
        last_day = today + pd.offsets.MonthEnd(0)
        days_left = (last_day - today).days + 1  # Including today until end of month
        return self.predict_next_n_days(days_left)

    def predict_by_date_range(self, start_date, end_date):
        return self.predict_range(pd.to_datetime(start_date), pd.to_datetime(end_date))

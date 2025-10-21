import pandas as pd
from datetime import datetime,timedelta
from ..algorithms.predictions_algorithm import TicketVolumePredictor

class PredictionService:
    """
    Service layer responsible for interacting with the TicketVolumePredictor model.
    This acts as a bridge between the Flask routes and the ML backend.
    """

    def __init__(self, csv_path="data/tickets_dataset_datetime_converted.csv"):
        """
        Initialize the prediction service and load the model.
        Args:
            csv_path (str): Path to the historical ticket data CSV file.
        """
        try:
            self.predictor = TicketVolumePredictor(csv_path)
        except Exception as e:
            raise

    def predict_tickets(self, start_date: datetime, end_date: datetime):
        """
        Predict ticket volumes for a specific date range using the backend model.
        Args:
            start_date (datetime): Start date for prediction range.
            end_date (datetime): End date for prediction range.
        Returns:
            list[dict]: A list of predictions with dates and predicted volumes.
        """
        try:
            predictions_df = self.predictor.predict_by_date_range(start_date, end_date)
            predictions = predictions_df.to_dict(orient="records")
            return predictions
        except Exception as e:
            raise

    def predicting_next_n_days(self, days: int = 7):
        """
        Predict ticket volume for the next 'n' days.
        Args:
            days (int): Number of days to predict forward.
        Returns:
            list[dict]: Predictions.
        """
        try:
            predictions_df = self.predictor.predict_next_n_days(days)
            predictions = predictions_df.to_dict(orient="records")
            return predictions
        except Exception as e:
            raise

    def predicting_this_week(self):
        """
        Shortcut method to predict this week's tickets.
        Returns:
            list[dict]: Predictions for the next 7 days.
        """
        try:
            predictions_df = self.predictor.predict_this_week()
            predictions = predictions_df.to_dict(orient="records")
            return predictions
        except Exception as e:
            raise

    def predicting_this_month(self):
        """
        Shortcut method to predict ticket volume for this month.
            Returns:
                list[dict]: Predictions for the next 30 days.
            """
        try:
            predictions_df = self.predictor.predict_this_month()
            predictions = predictions_df.to_dict(orient="records")
            return predictions
        except Exception as e:
                raise

    def predicting_next_2_days(self):
        """
        Predict ticket volume for next 2 days.
        Returns:
            list[dict]: Predictions for next 2 days.
        """
        try:
            predictions_df = self.predictor.predict_next_2_days()
            predictions = predictions_df.to_dict(orient="records")
            return predictions
        except Exception as e:
            raise

    def predicting_next_7_days(self):
        """
        Predict ticket volume for next 7 days.
        Returns:
            list[dict]: Predictions for next 7 days.
        """
        try:
            predictions_df = self.predictor.predict_next_7_days()
            predictions = predictions_df.to_dict(orient="records")
            return predictions
        except Exception as e:
            raise

    def predicting_next_30_days(self):
        """
        Predict ticket volume for next 30 days.
        Returns:
            list[dict]: Predictions for next 30 days.
        """
        try:
            predictions_df = self.predictor.predict_next_30_days()
            predictions = predictions_df.to_dict(orient="records")
            return predictions
        except Exception as e:
            raise

    def get_health_status(self):
        """
        Check the health and readiness of the prediction service.
        Returns:
            dict: Health status information.
        """
        try:
            return {
                "status": "ready" if self.predictor else "not_ready",
                "last_trained": str(self.predictor.daily_df['start_date'].max()) if self.predictor else None,
                "model_info": {
                    "features": len(self.predictor.features),
                    "data_points": len(self.predictor.daily_df) if self.predictor else 0
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def predicting_tomorrow(self):
        """
        Predict ticket volume for tomorrow.
        Returns:
            list[dict]: Predictions for tomorrow.
        """
        try:
            tomorrow = datetime.now().date() + timedelta(days=1)
            predictions_df = self.predictor.predict_tomorrow()
            predictions = predictions_df.to_dict(orient="records")
            return predictions
        except Exception as e:
            raise
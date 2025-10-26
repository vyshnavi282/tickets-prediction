from datetime import datetime
from typing import List, Dict
# Import the core prediction algorithm class from the relative path
from ..algorithms.predictions_algorithm import TicketVolumePredictor

# Define a service class to manage ticket volume predictions
class PredictionService:
    # Constructor for the service, initializes the predictor
    def __init__(self, csv_path="data/tickets_dataset_datetime_converted.csv"):
        try:
            # Print a message indicating the predictor is being initialized
            print("****** Initializing TicketVolumePredictor *****")
            # Instantiate the prediction algorithm class
            self.predictor = TicketVolumePredictor(csv_path)
        except Exception as e:
            # Re-raise any exception that occurs during initialization
            raise

    # Private helper method to execute a prediction function and convert results
    def _predict_helper(self, prediction_func) -> List[Dict]:
        try:
            # Execute the prediction function (which should return a pandas DataFrame)
            df = prediction_func()
            # Convert the resulting DataFrame to a list of dictionaries (records)
            return df.to_dict(orient='records')
        except Exception as e:
            # Re-raise any exception that occurs during prediction or conversion
            raise

    # Public method to predict ticket volume for a specified date range
    def predict_tickets(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        # Use the helper method with a lambda function that calls the predictor's range prediction
        return self._predict_helper(lambda: self.predictor.predict_range(start_date,
                                                                         end_date))

    # Public method to retrieve the service's health and model status
    def get_health_status(self) -> Dict:
        try:
            # Determine the service status based on whether the predictor is initialized
            status = 'ready' if self.predictor else 'not_ready'
            # Get the date of the latest data point used for training, if available
            last_trained = str(self.predictor.daily_df['start_date'].max(
            )) if self.predictor else None
            # Gather information about the model's features and data points
            model_info = {
                'features': len(self.predictor.features) if self.predictor else 0,
                'data_points': len(self.predictor.daily_df) if self.predictor
                and hasattr(self.predictor, 'daily_df') else 0
            }
            # Return a dictionary containing the comprehensive health status
            return {
                'status': status,
                'last_trained': last_trained,
                'model_info': model_info,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            # If an error occurs during status check, return an error dictionary
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

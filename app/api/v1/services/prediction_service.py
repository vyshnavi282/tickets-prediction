from datetime import datetime
from typing import List, Dict
from ..algorithms.predictions_algorithm import TicketVolumePredictor

class PredictionService:
    def __init__(self, csv_path="data/tickets_dataset_datetime_converted.csv"):
        try:
            print("****** Initializing TicketVolumePredictor *****")
            self.predictor = TicketVolumePredictor(csv_path)
        except Exception as e:
            raise

    def _predict_helper(self, prediction_func) -> List[Dict]:
        try:
            df = prediction_func()
            return df.to_dict(orient='records')
        except Exception as e:
            raise

    def predict_tickets(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        return self._predict_helper(lambda: self.predictor.predict_range(start_date,
                                                                        end_date))

    def get_health_status(self) -> Dict:
        try:
            status = 'ready' if self.predictor else 'not_ready'
            last_trained = str(self.predictor.daily_df['start_date'].max(
            )) if self.predictor else None
            model_info = {
                'features': len(self.predictor.features) if self.predictor else 0,
                'data_points': len(self.predictor.daily_df) if self.predictor 
                and hasattr(self.predictor, 'daily_df') else 0
            }
            return {
                'status': status,
                'last_trained': last_trained,
                'model_info': model_info,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

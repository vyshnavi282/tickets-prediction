from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
from ..algorithms.predictions_algorithm import TicketVolumePredictor

class PredictionService:
    def __init__(self, csv_path="data/tickets_dataset_datetime_converted.csv"):
        try:
            self.predictor = TicketVolumePredictor(csv_path)
        except Exception as e:
            raise

    def _predict_helper(self, prediction_func) -> List[Dict]:
        try:
            predictions_df = prediction_func()
            return predictions_df.to_dict(orient='records')
        except Exception as e:
            raise

    def predict_tickets(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        return self._predict_helper(lambda: self.predictor.predict_by_date_range(start_date, end_date))

    def predicting(self, prediction_type: str, days: Optional[int] = None) -> List[Dict]:
        """
        General method to get predictions by type using switch-case (match).
        prediction_type examples: 'next_n_days', 'this_week', 'this_month', 'next_2_days', etc.
        If prediction_type requires days parameter (e.g. 'next_n_days'), days must be provided.
        """
        match prediction_type:
            case 'next_n_days':
                if days is None:
                    raise ValueError("Days parameter required for 'next_n_days'")
                return self._predict_helper(lambda: self.predictor.predict_next_n_days(days))
            case 'this_week':
                return self._predict_helper(self.predictor.predict_this_week)
            case 'this_month':
                return self._predict_helper(self.predictor.predict_this_month)
            case 'next_2_days':
                return self._predict_helper(self.predictor.predict_next_2_days)
            case 'next_7_days':
                return self._predict_helper(self.predictor.predict_next_7_days)
            case 'next_30_days':
                return self._predict_helper(self.predictor.predict_next_30_days)
            case 'tomorrow':
                return self._predict_helper(self.predictor.predict_tomorrow)
            case _:
                raise ValueError(f"Unsupported prediction type: {prediction_type}")

    def get_health_status(self) -> Dict:
        try:
            status = 'ready' if self.predictor else 'not_ready'
            last_trained = str(self.predictor.daily_df['start_date'].max()) if self.predictor else None
            model_info = {
                'features': len(self.predictor.features),
                'data_points': len(self.predictor.daily_df) if self.predictor else 0
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

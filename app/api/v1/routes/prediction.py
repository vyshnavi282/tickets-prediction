from flask import Blueprint, request, jsonify
from datetime import datetime
from functools import wraps
from ..services.prediction_service import PredictionService

prediction_routes = Blueprint('predictions', __name__)

try:
    prediction_service = PredictionService()
except Exception as e:
    prediction_service = None
    print(f"Error initializing PredictionService: {e}")

def check_service_initialized(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not prediction_service:
            return jsonify({'error': 'Prediction service not initialized', 'status': 'error'}), 503
        return func(*args, **kwargs)
    return wrapper

def make_response(data=None, error=None, status_code=200, days_predicted=None):
    if error:
        response = {'status': 'error', 'error': error}
    else:
        response = {
            'status': 'success',
            'data': data or {},
            'metadata': {
                'days_predicted': days_predicted,
                'timestamp': datetime.now().isoformat()
            }
        }
    return jsonify(response), status_code

@prediction_routes.route('/health', methods=['GET'])
def health_check():
    if not prediction_service:
        return jsonify({'status': 'unhealthy', 'error': 'Prediction service not initialized'}), 503
    try:
        status = prediction_service.get_health_status()
        return jsonify({'status': 'healthy', 'service_status': status, 'timestamp': datetime.now().isoformat()})
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e), 'timestamp': datetime.now().isoformat()}), 500

@prediction_routes.route('/predictions', methods=['GET'])
@check_service_initialized
def predict_date_range():
    start_date = request.args.get('startdate')
    end_date = request.args.get('enddate')

    if not start_date or not end_date:
        return make_response(
            error='Both startdate and enddate query parameters are required. Example: /api/v1/predictions?startdate=2025-10-17&enddate=2025-10-24',
            status_code=400
        )

    # Parse and validate date format
    try:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError:
        return make_response(
            error='Invalid date format. Use YYYY-MM-DD. Example: /api/v1/predictions?startdate=2025-10-17&enddate=2025-10-24',
            status_code=400
        )

    if end_dt < start_dt:
        return make_response(error='End date must be after start date', status_code=400)

    date_diff = (end_dt - start_dt).days + 1
    if date_diff > 90:
        return make_response(error='Prediction range cannot exceed 90 days', status_code=400)

    try:
        predictions = prediction_service.predict_tickets(start_dt, end_dt)
        if not predictions:
            return make_response(error='No predictions available for the specified date range', status_code=404)

        return make_response(
            data={'start_date': start_date, 'end_date': end_date, 'predictions': predictions},
            days_predicted=date_diff
        )
    except Exception as e:
        return make_response(error=f'Internal server error: {str(e)}', status_code=500)

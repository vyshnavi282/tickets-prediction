from flask import Blueprint, request, jsonify
from datetime import datetime
from functools import wraps
from ..services.prediction_service import PredictionService

prediction_routes = Blueprint('predictions', __name__)

# Initialize prediction service globally, log errors instead of failing silently
try:
    prediction_service = PredictionService()
except Exception as e:
    prediction_service = None
    print(f"Error initializing PredictionService: {e}")

# Decorator to check if service initialized
def check_service_initialized(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not prediction_service:
            return jsonify({'error': 'Prediction service not initialized', 'status': 'error'}), 503
        return func(*args, **kwargs)
    return wrapper

# Helper for uniform response with metadata
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
    """Check health status of the prediction service."""
    if not prediction_service:
        return jsonify({
            'status': 'unhealthy',
            'error': 'Prediction service not initialized'
        }), 503

    try:
        service_status = prediction_service.get_health_status()
        return jsonify({
            'status': 'healthy',
            'service_status': service_status,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@prediction_routes.route('/predictions', methods=['GET'])
@check_service_initialized
def predict_date_range():
    """
    Get predictions for a specific date range
    Endpoint: /api/v1/predictions?startdate=YYYY-MM-DD&enddate=YYYY-MM-DD
    """
    start_date = request.args.get('startdate')
    end_date = request.args.get('enddate')

    if not start_date or not end_date:
        return make_response(
            error='Both start date and end date parameters are required; example: /api/v1/predictions?startdate=2025-10-17&enddate=2025-10-24',
            status_code=400
        )

    try:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError:
        return make_response(
            error='Invalid date format. Use YYYY-MM-DD; example: /api/v1/predictions?startdate=2025-10-17&enddate=2025-10-24',
            status_code=400
        )

    if end_dt < start_dt:
        return make_response(error='End date must be after start date', status_code=400)

    date_diff = (end_dt - start_dt).days + 1
    if date_diff > 90:
        return make_response(error='Prediction range cannot exceed 90 days', status_code=400)

    predictions = prediction_service.predict_tickets(start_dt, end_dt)
    if not predictions:
        return make_response(error='No predictions available for the specified date range', status_code=404)

    data = {
        'start_date': start_date,
        'end_date': end_date,
        'predictions': predictions
    }
    return make_response(data=data, days_predicted=date_diff)



# Optional: Route to match the current POST method replaced by consolidated GET route above
@prediction_routes.route('/predictions', methods=['POST'])
@check_service_initialized
def predicted_next_n_days_post():
    """
    POST version for next N days prediction to maintain compatibility.
    Body JSON: { "days": 10 }
    """
    data = request.get_json()
    if not data or 'days' not in data:
        return make_response(error="Missing 'days' field in request body", status_code=400)

    days = data['days']
    if not isinstance(days, int) or days < 1 or days > 90:
        return make_response(error='Days parameter must be an integer between 1 and 90', status_code=400)

    predictions = prediction_service.predicting('next_n_days', days)
    return make_response(data={'predictions': predictions}, days_predicted=days)

@prediction_routes.route('/predictions/<string:preset>', methods=['GET'])
@check_service_initialized
def predictions_preset(preset):
    """
    Get predictions for preset date ranges like 'this_week', 'this_month', 'next_2_days', 'tomorrow'.
    """
    preset_methods ={
    'this_week': lambda: prediction_service.predicting('this_week'),
    'this_month': lambda: prediction_service.predicting('this_month'),
    'next_2_days': lambda: prediction_service.predicting('next_2_days'),
    'next_7_days': lambda: prediction_service.predicting('next_7_days'),
    'next_30_days': lambda: prediction_service.predicting('next_30_days'),
    'tomorrow': lambda: prediction_service.predicting('tomorrow')
    }

    predictions = preset_methods[preset]()


    if preset not in preset_methods:
        return make_response(error='Invalid preset parameter', status_code=400)

    try:
        predictions = prediction_service.predicting(preset)
        days_map = {
            'this_week': 7,
            'this_month': 30,
            'next_2_days': 2,
            'next_7_days': 7,
            'next_30_days': 30,
            'tomorrow': 1
        }
        return make_response(data={'predictions': predictions}, days_predicted=days_map[preset])
    except Exception as e:
        return make_response(error=str(e), status_code=500)

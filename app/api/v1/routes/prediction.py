from flask import Blueprint, request, jsonify
from datetime import datetime, timedelta
from ..services.prediction_service import PredictionService

prediction_routes = Blueprint('predictions', __name__)

# Initialize prediction service as a global variable
try:
    prediction_service = PredictionService()
except Exception as e:
    prediction_service = None

@prediction_routes.route('/predictions', methods=['GET'])
def predict_date_range():
    """
    Get predictions for a specific date range
    Endpoint: /api/v1/predictions?startdate=YYYY-MM-DD&enddate=YYYY-MM-DD
    """
    try:
        # Check if prediction service is initialized
        if not prediction_service:
            return jsonify({
                'error': 'Prediction service is not initialized',
                'status': 'error'
            }), 503

        # Get and validate date parameters
        start_date = request.args.get('startdate')
        end_date = request.args.get('enddate')

        if not start_date or not end_date:
            return jsonify({
                'error': 'Both startdate and enddate parameters are required',
                'example': '/api/v1/predictions?startdate=2025-10-17&enddate=2025-10-24'
            }), 400

        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            return jsonify({
                'error': 'Invalid date format. Use YYYY-MM-DD',
                'example': '/api/v1/predictions?startdate=2025-10-17&enddate=2025-10-24'
            }), 400

        # Validate date range
        if end_dt < start_dt:
            return jsonify({
                'error': 'End date must be after start date'
            }), 400

        # Validate maximum prediction range (e.g., 90 days)
        date_diff = (end_dt - start_dt).days + 1
        if date_diff > 90:
            return jsonify({
                'error': 'Prediction range cannot exceed 90 days'
            }), 400

        # Get predictions from service
        predictions = prediction_service.predict_tickets(
            start_date=start_dt,
            end_date=end_dt
        )

        if not predictions:
            return jsonify({
                'error': 'No predictions available for the specified date range'
            }), 404

        return jsonify({
            'status': 'success',
            'data': {
                'start_date': start_date,
                'end_date': end_date,
                'predictions': predictions,
                'metadata': {
                    'days_predicted': date_diff,
                    'timestamp': datetime.now().isoformat()
                }
            }
        })

    except Exception as e:
        return jsonify({
            'error': 'Internal server error occurred',
            'status': 'error',
            'details': str(e)
        }), 500

@prediction_routes.route('/health', methods=['GET'])
def health_check():
    """
    Check the health status of the prediction service
    Endpoint: /api/v1/health
    """
    try:
        if not prediction_service:
            return jsonify({
                'status': 'unhealthy',
                'error': 'Prediction service not initialized'
            }), 503

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

@prediction_routes.route('/predictions/next_week', methods=['GET'])
def predicted_next_7_days():
    """
    Get predictions for the next 7 days
    Endpoint: /api/v1/predictions/next_week
    """
    try:
        if not prediction_service:
            return jsonify({
                'error': 'Prediction service not initialized',
                'status': 'error'
            }), 503

        predictions = prediction_service.predicting_next_7_days()
        return jsonify({
            'status': 'success',
            'data': {
                'predictions': predictions,
                'metadata': {
                    'days_predicted': 7,
                    'timestamp': datetime.now().isoformat()
                }
            }
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500
@prediction_routes.route('/predictions/next', methods=['POST'])
def predicted_next_n_days():
    """
    Get ticket volume predictions for the next 'days' days
    Endpoint: /api/v1/predictions/next
    Request body example:
    {
        "days": 10
    }
    """
    try:
        if not prediction_service:
            return jsonify({
                'error': 'Prediction service not initialized',
                'status': 'error'
            }), 503

        data = request.get_json()  # Get JSON body from request

        if not data or 'days' not in data:
            return jsonify({
                'error': "Missing 'days' field in request body"
            }), 400

        days = data['days']

        if not isinstance(days, int) or days <= 0 or days > 90:
            return jsonify({
                'error': 'Days parameter must be an integer between 1 and 90'
            }), 400

        predictions = prediction_service.predicting_next_n_days(days)
        return jsonify({
            'status': 'success',
            'data': {
                'predictions': predictions,
                'metadata': {
                    'days_predicted': days,
                    'timestamp': datetime.now().isoformat()
                }
            }
        }), 200
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500
    
@prediction_routes.route('/predictions/next_month', methods=['GET'])
def predicted_next_30_days():
    """
    Get ticket volume predictions for the next 30 days
    Endpoint: /api/v1/predictions/next_month
    Note: Returns predictions for exactly 30 days from current date
    """
    try:
        if not prediction_service:
            return jsonify({
                'error': 'Prediction service not initialized',
                'status': 'error'
            }), 503

        predictions = prediction_service.predicting_next_30_days()
        return jsonify({
            'status': 'success',
            'data': {
                'predictions': predictions,
                'metadata': {
                    'days_predicted': 30,
                    'timestamp': datetime.now().isoformat()
                }
            }
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@prediction_routes.route('/predictions/this_month', methods=['GET'])
def predicted_this_month():
    """
    Get ticket volume predictions for the this 30 days (this month)
    Endpoint: /api/v1/predictions/this_month
    """
    try:
        if not prediction_service:
            return jsonify({
                'error': 'Prediction service not initialized',
                'status': 'error'
            }), 503

        predictions = prediction_service.predicting_this_month()
        return jsonify({
            'status': 'success',
            'data': {
                'predictions': predictions,
                'metadata': {
                    'days_predicted': 30,
                    'timestamp': datetime.now().isoformat()
                }
            }
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500
    
@prediction_routes.route('/predictions/this_week', methods=['GET'])
def predicted_this_week():
    """
    Get predictions for this week
    Endpoint: /api/v1/predictions/this_week
    """
    try:
        if not prediction_service:
            return jsonify({
                'error': 'Prediction service not initialized',
                'status': 'error'
            }), 503

        predictions = prediction_service.predicting_this_week()
        return jsonify({
            'status': 'success',
            'data': {
                'predictions': predictions,
                'metadata': {
                    'days_predicted': 7,
                    'timestamp': datetime.now().isoformat()
                }
            }
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500
    
@prediction_routes.route('/predictions/next_2_days', methods=['GET'])
def predicted_next_2_days():
    """
    Get predictions for the next 2 days
    Endpoint: /api/v1/predictions/next_2_days
    """
    try:
        if not prediction_service:
            return jsonify({
                'error': 'Prediction service not initialized',
                'status': 'error'
            }), 503

        predictions = prediction_service.predicting_next_2_days()
        return jsonify({
            'status': 'success',
            'data': {
                'predictions': predictions,
                'metadata': {
                    'days_predicted': 2,
                    'timestamp': datetime.now().isoformat()
                }
            }
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@prediction_routes.route('/predictions/tomorrow', methods=['GET'])
def predicted_tomorrow():
    """
    Get predictions for tomorrow
    Endpoint: /api/v1/predictions/tomorrow
    """
    try:
        if not prediction_service:
            return jsonify({
                'error': 'Prediction service not initialized',
                'status': 'error'
            }), 503

        predictions = prediction_service.predicting_tomorrow()
        return jsonify({
            'status': 'success',
            'data': {
                'predictions': predictions,
                'metadata': {
                    'days_predicted': 1,
                    'timestamp': datetime.now().isoformat()
                }
            }
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500
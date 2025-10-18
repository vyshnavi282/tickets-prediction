from flask import Blueprint, request, jsonify
from datetime import datetime, timedelta
import logging
from ..services.prediction_service import PredictionService

logger = logging.getLogger(__name__)
prediction_routes = Blueprint('predictions', __name__)

# Initialize prediction service as a global variable
try:
    prediction_service = PredictionService()
except Exception as e:
    logger.error(f"Failed to initialize PredictionService: {str(e)}")
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
        logger.error(f"Error in prediction endpoint: {str(e)}")
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
        logger.error(f"Error in health check endpoint: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@prediction_routes.route('/predictions/week', methods=['GET'])
def predict_this_week():
    """
    Get predictions for the next 7 days
    Endpoint: /api/v1/predictions/week
    """
    try:
        if not prediction_service:
            return jsonify({
                'error': 'Prediction service not initialized',
                'status': 'error'
            }), 503

        predictions = prediction_service.predict_this_week()
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
        logger.error(f"Error in weekly prediction endpoint: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500
    
@prediction_routes.route('/predictions/next/<int:days>', methods=['GET'])
def predict_next_n_days(days):
    """
    Get ticket volume predictions for the next 'days' days
    Endpoint: /api/v1/predictions/next/<days>
    """
    try:
        if not prediction_service:
            return jsonify({
                'error': 'Prediction service not initialized',
                'status': 'error'
            }), 503

        if days <= 0 or days > 90:
            return jsonify({
                'error': 'Days parameter must be between 1 and 90'
            }), 400

        predictions = prediction_service.predict_next_n_days(days)
        return jsonify({
            'status': 'success',
            'data': {
                'predictions': predictions,
                'metadata': {
                    'days_predicted': days,
                    'timestamp': datetime.now().isoformat()
                }
            }
        })
    except Exception as e:
        logger.error(f"Error in next_n_days prediction endpoint: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@prediction_routes.route('/predictions/month', methods=['GET'])
def predict_this_month():
    """
    Get ticket volume predictions for the next 30 days (this month)
    Endpoint: /api/v1/predictions/month
    """
    try:
        if not prediction_service:
            return jsonify({
                'error': 'Prediction service not initialized',
                'status': 'error'
            }), 503

        predictions = prediction_service.predict_this_month()
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
        logger.error(f"Error in monthly prediction endpoint: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

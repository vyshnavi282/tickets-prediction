from flask import Blueprint, request, jsonify
from datetime import datetime, timedelta
import logging
from ...services.prediction_service import PredictionService

logger = logging.getLogger(__name__)
prediction_routes = Blueprint('predictions', __name__)
prediction_service = PredictionService()

@prediction_routes.route('/predictions', methods=['GET'])
def predict_date_range():
    """
    Get predictions for a specific date range
    Endpoint: /api/v1/predictions?startdate=YYYY-MM-DD&enddate=YYYY-MM-DD
    """
    try:
        # Get start_date and end_date from query parameters
        start_date = request.args.get('startdate')
        end_date = request.args.get('enddate')

        # Validate date parameters
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

        # Calculate number of days in the range
        date_diff = (end_dt - start_dt).days + 1

        # Get predictions from service
        predictions = prediction_service.predict_tickets(
            start_date=start_dt,
            end_date=end_dt
        )

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
            'error': str(e),
            'status': 'error'
        }), 500

@prediction_routes.route('/health', methods=['GET'])
def health_check():
    """
    Check the health status of the prediction service
    Endpoint: /api/v1/health
    """
    try:
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error in health check endpoint: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

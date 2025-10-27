from flask import Blueprint, request, jsonify
from datetime import datetime
from functools import wraps
# Import the prediction service class
from ..services.prediction_service import PredictionService

# Create a Flask Blueprint for prediction routes
prediction_routes = Blueprint('predictions', __name__)

# Attempt to initialize the global prediction service instance
try:
    prediction_service = PredictionService()
except Exception as e:
    # If initialization fails, set the service to None and log the error
    prediction_service = None
    print(f"Error initializing PredictionService: {e}")

# Decorator to check if the prediction service was successfully initialized
def check_service_initialized(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Return a 503 error if the service instance is not available
        if not prediction_service:
            return jsonify({'error': 'Prediction service not initialized',
                            'status': 'error'}), 503
        return func(*args, **kwargs)
    return wrapper

# Helper function to standardize API response format
def make_response(data=None, error=None, status_code=200, days_predicted=None):
    if error:
        # Create an error response structure
        response = {'status': 'error', 'error': error}
    else:
        # Create a success response structure with data and metadata
        response = {
            'status': 'success',
            'data': data or {},
            'metadata': {
                'days_predicted': days_predicted,
                'timestamp': datetime.now().isoformat()
            }
        }
    # Return the JSON response and HTTP status code
    return jsonify(response), status_code

# Define the health check endpoint
@prediction_routes.route('/health', methods=['GET'])
def health_check():
    # Handle the case where the service failed to initialize
    if not prediction_service:
        return jsonify({'status': 'unhealthy',
                         'error': 'Prediction service not initialized'}), 503
    try:
        # Get detailed status from the prediction service
        status = prediction_service.get_health_status()
        # Return a detailed healthy status response
        return jsonify({'status': 'healthy', 'service_status': status,
                         'timestamp': datetime.now().isoformat()})
    except Exception as e:
        # Handle exceptions during the health check itself
        return jsonify({'status': 'unhealthy', 'error': str(e),
                         'timestamp': datetime.now().isoformat()}), 500

# Define the main prediction endpoint
@prediction_routes.route('/', methods=['GET'])
@check_service_initialized  # Ensure the service is ready before proceeding
def predict_date_range():
    # Get start and end dates from query parameters
    start_date = request.args.get('startdate')
    end_date = request.args.get('enddate')

    # Validate presence of required parameters
    if not start_date or not end_date:
        return make_response(
            error="""Both startdate and enddate query parameters are required.
              Example: /api/v1/predictions?startdate=2025-10-17&enddate=2025-10-24""",
            status_code=400
        )

    # Parse and validate date format
    try:
        starting_date = datetime.strptime(start_date, '%Y-%m-%d')
        ending_date = datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError:
        return make_response(
            error="""Invalid date format. Use YYYY-MM-DD.
              Example: /api/v1/predictions?startdate=2025-10-17&enddate=2025-10-24""",
            status_code=400
        )

    # Validate that end date is not before start date
    if ending_date < starting_date:
        return make_response(error='End date must be after start date', status_code=400)

    # Calculate the number of days in the range (inclusive)
    date_difference = (ending_date - starting_date).days + 1
    # Enforce a maximum prediction range limit
    if date_difference > 90:
        return make_response(error='Prediction range cannot exceed 90 days',
                              status_code=400)

    # Call the prediction service
    try:
        predictions = prediction_service.predict_tickets(starting_date, ending_date)
        # Handle case where the service returns no predictions
        if not predictions:
            return make_response(error='No predictions available for the' \
            'specified date range', status_code=404)

        # Return the final success response with predictions
        return make_response(
            data={'start_date': start_date, 'end_date': end_date,
            'predictions': predictions},
            days_predicted=date_difference
        )
    except Exception as e:
        # Handle internal errors during prediction
        return make_response(error=f'Internal server error: {str(e)}', status_code=500)
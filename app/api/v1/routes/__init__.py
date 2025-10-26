from .prediction import prediction_routes
from .prediction_agent import prediction_agent_bp

# Function to register all blueprints (sets of routes) with the Flask application
def register_routes(app):
    # Print a status message indicating the function is executed
    print("====== 2 Inside routes function")
    # Register the standard prediction routes blueprint
    # Routes in this blueprint will be prefixed with /api/v1/predictions
    app.register_blueprint(prediction_routes, url_prefix="/api/v1/predictions")
    # Register the AI agent prediction routes blueprint
    # Routes in this blueprint will be prefixed with /api/v1/agent-predictions
    app.register_blueprint(prediction_agent_bp, url_prefix="/api/v1/agent-predictions")
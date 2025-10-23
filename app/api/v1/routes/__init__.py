from .prediction import prediction_routes
from .prediction_agent import prediction_agent_bp

def register_routes(app):
    print("====== 2 Inside routes function")
    app.register_blueprint(prediction_routes, url_prefix="/api/v1/predictions")
    app.register_blueprint(prediction_agent_bp, url_prefix="/api/v1/agent-predictions")
    
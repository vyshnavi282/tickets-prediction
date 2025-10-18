from .prediction import prediction_routes

def register_routes(app):
    print("====== 2 Inside routes function")
    app.register_blueprint(prediction_routes, url_prefix="/api/v1")

    
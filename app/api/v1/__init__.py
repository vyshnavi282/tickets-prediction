from flask import Flask
# Import the function to register routes from the relative path '.routes'
from .routes import register_routes

import logging
# Configure basic logging settings (e.g., set the minimum level to INFO)
logging.basicConfig(level=logging.INFO)

# Define the application factory function
def create_app():
    # Create the Flask application instance
    app = Flask(__name__)

    # Log an informational message indicating entry into the function
    logging.info("Inside create_app function")

    # Register the application's routes by calling the imported function
    register_routes(app)

    # Return the configured Flask application object
    return app


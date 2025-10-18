from flask import Flask
from .routes import register_routes
 
import logging
logging.basicConfig(level=logging.INFO)

def create_app():
    app = Flask(__name__)
    logging.info("Inside create_app function")
    register_routes(app)
    return app


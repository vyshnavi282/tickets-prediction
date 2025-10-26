# Import the application factory function from the specified module path
from app.api.v1 import create_app

# Call the factory function to create and configure the Flask application instance
app = create_app()

# Standard Python idiom to ensure the server only runs when the script is executed directly
if __name__ == "__main__":
    # Run the Flask development server
    # debug=True enables the interactive debugger and reloader
    app.run(debug=True)
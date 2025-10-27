from flask import Blueprint, request, jsonify
import asyncio
# Import the asynchronous function that runs the AI agent
from ..services.prediction_agent_service import get_predictions_with_agent

# Create a Flask Blueprint for the prediction agent routes
prediction_agent_bp = Blueprint('prediction_agent', __name__)

# Define the route for making predictions using the AI agent
@prediction_agent_bp.route('/', methods=['POST'])
def agent_predictions():
    # Get the JSON data sent in the POST request body
    data = request.json

    # Attempt to process the request using the asynchronous agent function
    try:
        # The agent function is async, so we run it in the event loop using asyncio.run
        # We pass the user's query from the JSON payload to the agent service
        result = asyncio.run(get_predictions_with_agent(data.get("query")))

        # Check if the result from the agent function is an error dictionary
        if isinstance(result, dict) and "error" in result:
            # Return an internal server error if the agent itself reported an error
            return jsonify({"error": result["error"]}), 500

        # Return the final human-readable summary from the agent as a success response
        return jsonify({"result": result}), 200

    except Exception as e:
        # Handle exceptions that occur during the asyncio or function execution
        return jsonify({"error": f"Request failed: {str(e)}"}), 500
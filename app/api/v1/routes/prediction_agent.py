from flask import Blueprint, request, jsonify
import asyncio
from ..services.prediction_agent_service import get_predictions_with_agent

prediction_agent_bp = Blueprint('prediction_agent', __name__)

@prediction_agent_bp.route('/', methods=['POST'])
def agent_predictions():
    #print("@@@@@@@@@@@ at req body")
    #print(request.json)
    data = request.json
    # start_date = request.args.get('startdate')
    # end_date = request.args.get('enddate')
    # req body query

    # if not start_date or not end_date:
    #     return jsonify({"error": "startdate and enddate parameters required"}), 400

    try:
        result = asyncio.run(get_predictions_with_agent(data.get("query")))
        if isinstance(result, dict) and "error" in result:
            return jsonify({"error": result["error"]}), 500
        return jsonify({"result": result}), 200
    except Exception as e:
        return jsonify({"error": f"Request failed: {str(e)}"}), 500


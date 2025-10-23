import asyncio
import httpx  # an async http client
import os
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = "http://127.0.0.1:5000/api/v1"

async def call_prediction_api(start_date: str, end_date: str) -> dict:
    """Call the base prediction API endpoint with start and end date query parameters."""
    url = f"{API_BASE_URL}/predictions/"  # Base URL without /predictions/ path
    params = {"startdate": start_date, "enddate": end_date}
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            try:
                return response.json()
            except Exception as e:
                return {"error": f"Failed to parse JSON response: {str(e)}",
                         "raw_response": response.text}
    except Exception as e:
        return {"error": f"API request failed: {str(e)}"}

async def prediction_tool(task: str, start_date: str = None, end_date: str = None) -> str:
    """Fetch prediction data and return a human-readable English summary string."""
    if not start_date or not end_date:
        return "Error: start_date and end_date are required."

    response = await call_prediction_api(start_date, end_date)

    if "error" in response:
        return f"Error from API call: {response['error']}"

    if response.get("status") != "success":
        return "API response status was not successful."

    predictions = response["data"]["predictions"]
    volumes = [p["predicted_ticket_volume"] for p in predictions]

    avg_volume = sum(volumes) / len(volumes)
    max_volume = max(volumes)
    min_volume = min(volumes)

    max_day = next(p for p in predictions if p["predicted_ticket_volume"] == max_volume)
    min_day = next(p for p in predictions if p["predicted_ticket_volume"] == min_volume)

    trend = "increasing" if volumes[-1] > volumes[0] else "decreasing"
    percent_change = ((volumes[-1] - volumes[0]) / volumes[0]) * 100 if volumes[0] != 0 else 0

    # Build human-readable summary
    summary = (
        f"Ticket volume forecast from {response['data']['start_date']} to {response['data']['end_date']} shows a {trend} trend. "
        f"The average daily ticket volume is {avg_volume:.2f}. "
        f"The highest predicted ticket volume is on {max_day['date']} with {max_volume:.2f} tickets, "
        f"while the lowest is on {min_day['date']} with {min_volume:.2f} tickets. "
    )
    
    if abs(percent_change) > 100:
        if percent_change > 0:
            change_desc = "more than doubled"
        else:
            change_desc = "less than halved"
        summary += f"Overall, the ticket volume has {change_desc} over the period."
    else:
        sign = "increased" if percent_change > 0 else "decreased"
        summary += f"This represents a {abs(percent_change):.2f}% {sign} during the period."
    print("Processed tool call result into human-readable summary.")
    return summary


async def get_predictions_with_agent(start_date: str, end_date: str) -> dict:
    """Get prediction data by running an AI agent that hits the Flask API."""
    print("Starting get_predictions_with_agent...")
    try:
        model_client = OpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
        agent = AssistantAgent(
        name="PredictionAgent",
        model_client=model_client,
        tools=[prediction_tool],
        system_message=(
               """ You are a helpful prediction analyst that explains ticket volume forecasts in clear, natural English.

            When analyzing ticket volume predictions:
            1. Begin with a short overview of the forecast period (mention the start and end dates).
            2. State the average daily ticket volume rounded to 2 decimal places.
            3. Identify which day has the highest predicted volume and which has the lowest.
            4. Describe the overall trend (e.g., increasing, decreasing, or stable).
            5. Explain any noticeable patterns or shifts in simple terms.
            6. If the prediction shows a significant percentage change (over 100%), explain it as “more than doubled” or similar plain phrasing.

            Important instructions:
            - Do NOT return the output in JSON or structured data.
            - Write only a coherent, human-readable narrative summary.
            - Make numbers easy to understand (round to 2 decimal places).
            - Keep the tone professional but conversational."""
            )
        )

        summary = await prediction_tool("get_predictions", start_date, end_date)

        agent_response = await agent.run(task = f"""Get and process predictions for {start_date} to
        {summary} for the data you get out of tool call and provide a 
        clear summary of the trends and patterns. Make it in human readable format. 
        You must give insights -- not the json data as is""")
 
        await model_client.close()
        print("############################Agent response received")
 
        print("response", agent_response)
        if hasattr(agent_response, 'messages') and agent_response.messages:
            return agent_response.messages[-1].content
        elif hasattr(agent_response,'content'):
            return agent_response.content
        else:
            return str(agent_response)
    except Exception as e:
        return f"Error getting weather with agent: {str(e)}"

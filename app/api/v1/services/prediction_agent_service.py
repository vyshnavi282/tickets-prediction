import asyncio
import httpx  # an async http client for making API requests
import os
from datetime import date
# Import the necessary components for the AutoGen agent framework
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

# Load environment variables (e.g., OPENAI_API_KEY) from a .env file
load_dotenv()

# Define the base URL for the local Flask API
API_BASE_URL = "http://127.0.0.1:5000/api/v1"

async def call_prediction_api(start_date: str, end_date: str) -> dict:
    """Call the base prediction API endpoint with start and end date query parameters."""
    # Construct the full URL for the predictions endpoint
    url = f"{API_BASE_URL}/predictions/"
    # Define the query parameters
    params = {"startdate": start_date, "enddate": end_date}
    try:
        # Use an asynchronous HTTP client to make the GET request
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            response = await client.get(url, params=params)
            # Raise an exception for bad status codes (4xx or 5xx)
            response.raise_for_status()
            try:
                # Attempt to parse the response as JSON
                return response.json()
            except Exception as e:
                # Handle JSON parsing errors
                return {"error": f"Failed to parse JSON response: {str(e)}",
                          "raw_response": response.text}
    except Exception as e:
        # Handle general API request errors (e.g., connection issues, timeouts)
        return {"error": f"API request failed: {str(e)}"}

async def prediction_tool(start_date:str = None, end_date: str = None)->str:
    """A tool function for the agent to call, which fetches and summarizes ticket predictions."""
    # Ensure both dates are provided
    if not start_date or not end_date:
        return "Error: start_date and end_date are required."

    # Call the actual prediction API
    response = await call_prediction_api(start_date, end_date)

    # Handle API errors
    if "error" in response:
        return f"Error from API call: {response['error']}"

    # Handle unsuccessful API status
    if response.get("status") != "success":
        return "API response status was not successful."

    # Extract the list of predictions and ticket volumes
    predictions = response["data"]["predictions"]
    volumes = [p["predicted_ticket_volume"] for p in predictions]

    # Calculate key statistical metrics
    avg_volume = sum(volumes) / len(volumes)
    max_volume = max(volumes)
    min_volume = min(volumes)

    # Find the data points corresponding to the max and min volumes
    max_day = next(p for p in predictions if p["predicted_ticket_volume"] == max_volume)
    min_day = next(p for p in predictions if p["predicted_ticket_volume"] == min_volume)

    # Determine the overall trend and percentage change
    trend = "increasing" if volumes[-1] > volumes[0] else "decreasing"
    # Calculate percentage change from start to end, handling division by zero
    percent_change = ((volumes[-1]
                      - volumes[0]) / volumes[0]) * 100 if volumes[0] != 0 else 0

    # Build a concise, human-readable summary string
    summary = (
        f"Ticket volume forecast from {response['data']
        ['start_date']} to {response['data']['end_date']} shows a {trend} trend."
        f"The average daily ticket volume is {avg_volume:.2f}. "
        f"The highest predicted ticket volume is on {max_day['date']
                                                     } with {max_volume:.2f} tickets,"
        f"while the lowest is on {min_day['date']} with {min_volume:.2f} tickets. "
    )
    
    # Add a specific description for large percentage changes
    if abs(percent_change) > 100:
        if percent_change > 0:
            change_desc = "more than doubled"
        else:
            change_desc = "less than halved"
        summary += f"Overall, the ticket volume has {change_desc} over the period."
    else:
        # Add the standard percentage change description
        sign = "increased" if percent_change > 0 else "decreased"
        summary += f"This represents a {abs(percent_change):.2f}% {
            sign} during the period."
    
    # Return the summary for the agent to incorporate into its final response
    return summary

async def get_predictions_with_agent(query: str) -> dict:
    """Get prediction data by running an AI agent that hits the Flask API using a tool."""
    try:
        # Get today's date to provide context to the agent
        today = date.today()
        # Initialize the model client for communication with the LLM
        model_client = OpenAIChatCompletionClient(
            model="gpt-5",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize the AssistantAgent with tools and a system message
        agent = AssistantAgent(
        name="PredictionAgent",
        model_client=model_client,
        tools=[prediction_tool],  # Register the prediction tool
        system_message=(
               f""" You are a helpful prediction analyst that explains
               ticket volume forecasts in clear and think creative , natural English.
               Today is {today}.
               From the query extract start date and end date properly  to call the tool.
               Call the prediction_tool to get the answer for the query.
               Remember the tool needs to be called with start_date and end_date
               in "YYYY-MM-DD".
               You must pass the neccesary parameters to call the tool.
               When analyzing ticket volume predictions:
               1. Begin with a short overview of the forecast period
                  (mention the start and end dates).
               2. State the average daily ticket volume rounded to 2 decimal places.
               3. Identify which day has the highest predicted volume and
                  which has the lowest.
               4. Describe the overall trend (e.g., increasing, decreasing, or stable).
               5. Explain any noticeable patterns or shifts in simple terms.
               6. If the prediction shows a significant percentage change
                  (over 100%), explain it as “more than doubled” or similar plain
                  phrasing.

               Important instructions:
               - Do NOT return the output in JSON or structured data.
               - Write only a coherent, human-readable narrative summary.
               - Make numbers easy to understand (round to 2 decimal places).
               - Keep the tone professional but conversational."""
               )
        )

        # Run the agent with the user's query as the task
        agent_response = await agent.run(task = f"""Predict the trend
            for the query: {query}. If the query is invalid, ask user for more clarifications
        """)
    
        # Close the model client connection
        await model_client.close()
    
        # Extract the final content from the agent's response object
        if hasattr(agent_response, 'messages') and agent_response.messages:
            return agent_response.messages[-1].content
        elif hasattr(agent_response,'content'):
            return agent_response.content
        else:
            return str(agent_response)
    except Exception as e:
        # Handle errors during agent initialization or execution
        return f"Error getting with agent: {str(e)}"

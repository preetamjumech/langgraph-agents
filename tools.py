# Import the keys
import os
import requests
from typing import List, Literal
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import tool
from dotenv import load_dotenv


load_dotenv()

WEATHER_API_KEY = os.environ['WEATHER_API_KEY']
TAVILY_API_KEY = os.environ['TAVILY_API_KEY']
TOGETHER_API_KEY = os.environ['TOGETHER_API_KEY']


@tool
def get_weather(query: str) -> list:
    """Search weatherapi to get the current weather"""
    endpoint = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={query}"
    response = requests.get(endpoint)
    data = response.json()

    if data.get("location"):
        return data
    else:
        return "Weather Data Not Found"

@tool
def search_web(query: str) -> list:
    """Search the web for a query"""
    tavily_search = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=2, search_depth='advanced', max_tokens=1000)
    results = tavily_search.invoke(query)
    return results


llm = ChatGoogleGenerativeAI(
    google_api_key=os.getenv("GEMINI_API_KEY"),
    model="gemini-2.0-flash",
    max_output_tokens=2048,
    temperature=0.2,
)


tools = [search_web, get_weather]
llm_with_tools = llm.bind_tools(tools)

prompt = """
    Given only the tools at your disposal, mention tool calls for the following tasks:
    Do not change the query given for any search tasks
        1. What is the current weather in Bengaluru today
        2. Can you tell me about Kolkata
        3. Why is the sky blue?
    """

results = llm_with_tools.invoke(prompt)

print(results.tool_calls) # Only shows which tools the LLM wants to call, not their results

query = "What is the current weather in Bengaluru today"
response = llm.invoke(query)
print(response.content)



# [{'name': 'get_weather', 'args': {'query': 'Bengaluru'}, 'id': '11954c4f-2bbb-4b73-9f52-7d3a0f4c6f92', 'type': 'tool_call'}, {'name': 'search_web', 'args': {'query': 'Kolkata'}, 'id': '116cc209-23fc-4811-ae04-aa14e5f0045c', 'type': 'tool_call'}, {'name': 'search_web', 'args': {'query': 'Why is the sky blue?'}, 'id': '048592e5-0187-4975-9479-78b062fd11ae', 'type': 'tool_call'}]       
# I do not have access to real-time information, including live weather updates. To get the current weather in Bengaluru, I recommend checking a reliable weather app or website such as:

# *   **Google Weather:** Just type "weather in Bengaluru" into Google.
# *   **AccuWeather:** [https://www.accuweather.com/](https://www.accuweather.com/)
# *   **The Weather Channel:** [https://weather.com/](https://weather.com/)
# *   **India Meteorological Department (IMD):** [http://imd.gov.in/](http://imd.gov.in/)

# These sources will provide you with the most up-to-date information.
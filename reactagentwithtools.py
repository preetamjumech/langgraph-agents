import os
import requests
from typing import List
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain.agents import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()

WEATHER_API_KEY = os.environ['WEATHER_API_KEY']
TAVILY_API_KEY = os.environ['TAVILY_API_KEY']
TOGETHER_API_KEY = os.environ['TOGETHER_API_KEY']

@tool
def get_weather(query: str) -> dict:
    """Search weatherapi to get the current weather"""
    endpoint = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={query}"
    response = requests.get(endpoint)
    data = response.json()
    if data.get("location"):
        return data
    else:
        return {"error": "Weather Data Not Found"}

@tool
def search_web(query: str) -> list:
    """Search the web for a query"""
    tavily_search = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=2, search_depth='advanced', max_tokens=1000)
    results = tavily_search.invoke(query)
    return results

llm = ChatOpenAI(base_url="https://api.together.xyz/v1",
    api_key=TOGETHER_API_KEY,
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
)

tools = [search_web, get_weather]



# system prompt is used to inform the tools available to when to use each
# system_prompt = """Act as a helpful assistant.
#     Use the tools at your disposal to perform tasks as needed.
#         - get_weather: whenever user asks get the weather of a place.
#         - search_web: whenever user asks for information on current events or if you don't know the answer.
#     Use the tools only if you don't know the answer.
#     """

# we can initialize the agent using the llama3 model, tools, and system prompt.
agent = create_react_agent(model=llm, tools=tools )

# Let’s query the agent to see the result.
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "What are the recent news in India?")]}

print_stream(agent.stream(inputs, stream_mode="values"))
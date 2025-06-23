import os
import requests
from typing import List
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain.agents import tool, initialize_agent, AgentType
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

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)


prompt = """
    Given only the tools at your disposal, mention tool calls for the following tasks:
    Do not change the query given for any search tasks
        1. What is the current weather in Bengaluru today
        2. Can you tell me about Kolkata
        3. Why is the sky blue?
    """
results = agent.invoke(prompt)

print(results)
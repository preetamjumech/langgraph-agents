from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_community.tools.tavily_search import TavilySearchResults
import os
import requests
from langchain_openai import ChatOpenAI
from typing import Literal
from dotenv import load_dotenv
load_dotenv()

WEATHER_API_KEY = os.environ['WEATHER_API_KEY']
TAVILY_API_KEY = os.environ['TAVILY_API_KEY']
TOGETHER_API_KEY = os.environ['TOGETHER_API_KEY']


def get_weather(query: str) -> dict:
    """Search weatherapi to get the current weather"""
    endpoint = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={query}"
    response = requests.get(endpoint)
    data = response.json()
    if data.get("location"):
        return data
    else:
        return {"error": "Weather Data Not Found"}


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
tool_node = ToolNode(tools)
llm_with_tools = llm.bind_tools(tools)


def call_model(state: MessagesState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def call_tools(state: MessagesState) -> Literal["tools", END]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END


# initialize the workflow from StateGraph
workflow = StateGraph(MessagesState)

# add a node named LLM, with call_model function. This node uses an LLM to make decisions based on the input given
workflow.add_node("LLM", call_model)

# Our workflow starts with the LLM node
workflow.add_edge(START, "LLM")

# Add a tools node
workflow.add_node("tools", tool_node)

# Add a conditional edge from LLM to call_tools function. It can go tools node or end depending on the output of the LLM. 
workflow.add_conditional_edges("LLM", call_tools)

# tools node sends the information back to the LLM
workflow.add_edge("tools", "LLM")

agent = workflow.compile()


for chunk in agent.stream(
    {"messages": [("user", "Will it rain in Bengaluru today?")]},
    stream_mode="values",):
    chunk["messages"][-1].pretty_print()
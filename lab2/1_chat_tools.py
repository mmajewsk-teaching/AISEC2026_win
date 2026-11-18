# Task 1: chatbot with tool calls (LangChain + Ollama). Fill in every `...`, in this order:
#   1. build_llm: create the ChatOllama model
#   2. chat_turn: build the message list and return the answer text
#   3. __main__: read input, call chat_turn, save both messages to history
#   4. run it: you have a working chatbot (without tools yet)
#   5. tools: `today`, `multiply`, `get_weather` (description + missing code)
#   6. tool_calls: loop while the model asks for tools
#   7. uncomment every `# @TODO` block, run it and ask about the date, a multiplication and the weather
# `# @TODO` marks code that is commented out on purpose (with `#` or `"""`): uncomment it once you get to that step.
# Run: pixi run python lab2/1_chat_tools.py   (needs: ollama pull qwen2.5:7b)
# Hints with links and quotes for every blank: lab2/1_chat_tools_hints.md
# Allowed websites (documentation only):
#   https://docs.langchain.com/oss/python/langchain/models             chat models, invoke, bind_tools
#   https://docs.langchain.com/oss/python/langchain/messages           System/Human/AI/ToolMessage
#   https://docs.langchain.com/oss/python/langchain/tools              @tool, tool calls
#   https://docs.langchain.com/oss/python/integrations/chat/ollama     ChatOllama
#   https://reference.langchain.com/python/                            LangChain API reference
#   https://docs.ollama.com/                                           Ollama (incl. capabilities/tool-calling)
#   https://open-meteo.com/en/docs                                     weather API
#   https://open-meteo.com/en/docs/geocoding-api                       city name -> coordinates
#   https://requests.readthedocs.io/en/latest/                         HTTP requests
#   https://docs.python.org/3/                                         Python (input, datetime)

from datetime import date

import requests
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

CHAT_MODEL = "qwen2.5:7b"

SYSTEM_PROMPT = "You are a helpful assistant. Answer briefly. Use tools when they help."


# @TODO
# @tool(description=...)
def today() -> str:
    return ....strftime("%A %Y-%m-%d")


# @TODO
# @tool(description=...)
def multiply(a: int, b: int) -> int:
    return ...


# @TODO
# @tool(description=...)
def get_weather(city: str) -> dict:
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": ..., "count": 1},
        timeout=10,
    ).json()
    if not geo.get("results"):
        return {"error": f"city '{...}' not found"}
    place = geo["results"][0]

    weather = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "temperature_2m,apparent_temperature,precipitation,cloud_cover,wind_speed_10m",
        },
        timeout=10,
    ).json()
    current = weather["current"]
    units = weather["current_units"]
    return {
        "city": f"{place['name']}, {place.get('country', '')}",
        **{
            key: f"{current[key]} {units[key]}"
            for key in current
            if key not in ("time", "interval")
        },
    }


def build_llm(model: str = CHAT_MODEL) -> ChatOllama:
    return ...(model=..., temperature=0)


# @TODO
"""
def tool_calls(llm, messages: list, response, tools: dict):
    while response....:
        messages.append(response)
        for call in response....:
            print(f"   [tool] {call['name']}({call['args']})")
            messages.append(tools[call["name"]].invoke(call))
        response = llm.invoke(...)
    return response
"""


def chat_turn(llm, history: list, user_message: str, tools: dict) -> str:
    # First we add the system prompt
    # Then we add all of the message history
    # Then the new message
    messages = [..., ..., ...]
    response = llm.invoke(messages)

    # @TODO
    # response = tool_calls(llm, messages, response, tools)
    
    return response....


if __name__ == "__main__":
    tools = {}
    llm = build_llm()

    # @TODO
    # tools = {t.name: t for t in [today, multiply, get_weather]}
    # llm = llm.bind_tools(list(tools.values()))

    print(f"Chatting with {CHAT_MODEL}, tools: {list(tools)}. Type \\q to quit.\n")
    history = []
    while True:
        try:
            # we want this to get input from the terminal
            question = ...(">>: ").strip()
        except EOFError:
            break
        if question == "\\q":
            break
        if not question:
            continue

        answer = ...(llm, history, question, tools)
        print(f"\n{answer}\n")

        history += [...(question), ...(answer)]

    print("bye")

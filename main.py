from typing import Annotated, List
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages

import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")


# 1. Define the State
# The state is a dictionary that holds the data passed between nodes
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    api_call_count: int


# 2. Define the Tool
# A simple tool that simulates an external API call
@tool
def fake_weather_api(city: str) -> str:
    """Check the weather in a specified city"""
    if city == "Berlin":
        return "Sunny 22°C"
    else:
        return "Service temporarily not available"


tools = [fake_weather_api]

# 3. Define the LLM
# Initialize the language model
llm = ChatOpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY, model="microsoft/phi-4")
llm_with_tools = llm.bind_tools(tools)
print(llm.invoke("what is this?"))



# 4. Define Nodes
# Nodes are the functions or runnables that perform specific tasks
def call_model(state: AgentState):
    """Invokes the agent model to generate a response based on the current state."""
    print("---CALL AGENT---")
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def call_tool(state: AgentState):
    """Executes a tool call if present in the last message."""
    print("---CALL TOOL---")
    messages = state["messages"]
    last_message = messages[-1]
    print(last_message)
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return {"messages": []}
    
    tool_messages = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments
        if tool_name == "fake_weather_api":
            import json
            tool_output = fake_weather_api.run(json.loads(tool_args).get("city"))
        else:
            tool_output = "tool not found"
        tool_messages.append(
            BaseMessage(
                content=str(tool_output),
                name=tool_name,
                tool_call_id=tool_call.id,
            )
        )
    return {
        "messages": tool_messages,
        "api_call_count": state["api_call_count"] + 1
    }


# 5. Define Conditional Edges
# Conditional edges determine the flow between nodes
def should_continue(state: AgentState):
    """Determines whether the agent should continue or end"""
    print("---SHOULD CONTINUE---")
    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"
    else:
        return "end"


# 6. Build the Graph
# Create the state graph and add the nodes and edges
workflow = StateGraph(AgentState)

# Add nodes to the graph
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)

# Set the entry point
workflow.set_entry_point("agent")

# Add conditional edges
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)

workflow.add_edge("action", "agent")

# 7. Compile the graph
graph = workflow.compile()


# 8. Run the graph
# Input for the graph is a dictionary with a message
inputs = {"messages": [HumanMessage(content="How's the weather in Berlin today?")],
          "api_call_count": 0
          }

#graph.stream(inputs)
for output in graph.stream(inputs):
    print(output)
#     for key, value in output.items():
#         print(f"Output from node '{key}':")
#         print("---")
#         print(value)
#         print("\n---\n")

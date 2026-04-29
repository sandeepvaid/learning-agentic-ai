import os
from dotenv import load_dotenv

# Load API keys from the .env file
load_dotenv()
# Step 1: Define tools and model
from langchain.tools import tool
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0   
)


# Define tools
@tool
def execute_math_plan(steps: list[dict]) -> float:
    """Execute a plan of chained mathematical operations and return the final result.
    The steps should be a list of dictionaries, each containing:
    - step_id: string (e.g., "var1")
    - operation: string ("add", "multiply", or "divide")
    - a: float or string (a number or a previous step_id)
    - b: float or string (a number or a previous step_id)
    """
    variables = {}
    
    for step in steps:
        step_id = step["step_id"]
        op = step["operation"]
        a = step["a"]
        b = step["b"]
        
        # Resolve variables if the argument is a string (a previous step_id)
        val_a = variables[a] if isinstance(a, str) and a in variables else float(a)
        val_b = variables[b] if isinstance(b, str) and b in variables else float(b)
        
        if op == "add":
            res = val_a + val_b
        elif op == "multiply":
            res = val_a * val_b
        elif op == "divide":
            res = val_a / val_b
        else:
            raise ValueError(f"Unknown operation: {op}")
            
        variables[step_id] = res
        print(f"Executed {op}({val_a}, {val_b}) -> {step_id} = {res}")
        
    # The result of the plan is the result of the last step
    return variables[steps[-1]["step_id"]]

# Augment the LLM with tools
tools = [execute_math_plan]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

# Step 2: Define state

from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

# Step 3: Define model node
from langchain.messages import SystemMessage


def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""

    # print(state["messages"])
    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }


# Step 4: Define tool node

from langchain.messages import ToolMessage


def tool_node(state: dict):
    """Performs the tool call"""

    result = []
    # print(state["messages"])
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

# Step 5: Define logic to determine whether to end

from typing import Literal
from langgraph.graph import StateGraph, START, END


# Conditional edge function to route to the tool node or end based upon whether the LLM made a tool call
def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    # print(messages)
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END

# Step 6: Build agent

# Build workflow
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

# Compile the agent
agent = agent_builder.compile()


# Invoke
from langchain.messages import HumanMessage
messages = [HumanMessage(content="Add 3 and 4 and then multiply it by 4 and then fivide it by 9 and then give me the result.")]
messages = agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()
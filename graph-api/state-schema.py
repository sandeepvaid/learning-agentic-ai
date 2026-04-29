#The StateGraph class is the main graph class to use. This is parameterized by a user defined State object.


# 1. What is state
# It consist of schema and reducer functions that actually updates the states. 
# Each state is used as in input in the graph  

#How we cam create multiple schemas
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class InputState(TypedDict):
    user_input: str

class OutputState(TypedDict):
    graph_output: str

class OverallState(TypedDict):
    foo: str
    user_input: str
    graph_output: str

class PrivateState(TypedDict):
    bar: str

def node_1(state: InputState) -> OverallState:
    # Write to OverallState
    return {"foo": state["user_input"] + " name"}

def node_2(state: OverallState) -> PrivateState:
    # Read from OverallState, write to PrivateState
    return {"bar": state["foo"] + " is"}

def node_3(state: PrivateState) -> OutputState:
    # Read from PrivateState, write to OutputState
    return {"graph_output": state["bar"] + " Lance"}

builder = StateGraph(OverallState,input_schema=InputState,output_schema=OutputState)
#Creating the node first for a graph
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)
#Starting the graph workflow
builder.add_edge(START, "node_1")
# Adding edge to the graph via connecting nodes
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", "node_3")
# Adding the END node to the graph
builder.add_edge("node_3", END)

# Compile the graph
graph = builder.compile()
# Invoking the graph
result = graph.invoke({"user_input":"My"})
print(result)

# {'graph_output': 'My name is Lance'}

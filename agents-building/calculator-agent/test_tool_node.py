from langchain.messages import AIMessage

# Now we can use a standard Python import!
from calculator_agent_improve import tool_node

def run_test():
    """This function tests the tool_node directly without calling the LLM."""
    
    print("\n[TEST] Creating mock LLM output...")
    # 1. Create a mock tool call exactly like the LLM would generate
    mock_tool_call = {
        "name": "execute_math_plan",
        "args": {
            "steps": [
                {"step_id": "var1", "operation": "add", "a": 3, "b": 4},
                {"step_id": "var2", "operation": "multiply", "a": "var1", "b": 4},
                {"step_id": "var3", "operation": "divide", "a": "var2", "b": 9}
            ]
        },
        "id": "call_mock123"
    }
    
    # 2. Wrap it in an AIMessage (this simulates what llm_node returns)
    mock_ai_message = AIMessage(content="", tool_calls=[mock_tool_call])
    
    # 3. Create the mock state to pass into the tool_node
    mock_state = {"messages": [mock_ai_message]}
    
    print("[TEST] Passing mock state to tool_node...\n")
    # 4. Call the tool_node directly
    result_state = tool_node(mock_state)
    
    print("\n[TEST] --- Final Results from tool_node ---")
    for msg in result_state["messages"]:
        msg.pretty_print()

if __name__ == "__main__":
    run_test()

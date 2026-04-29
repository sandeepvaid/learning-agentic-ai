#using reducer to update the schema

from langchain_core.runnables import add
from typing import Annotated
from typing_extensions import TypedDict

class State(TypedDict):
    input1:str
    # here no reducer is attached , so values will be overwrriten always 
    input2:list[str]


# Example usage:    
# {"input1": 1, "input2": ["hi"]}. Let’s then assume the first Node returns {"input1": 2}. This is treated as an update to the state. Notice that the Node does not need to return the whole State schema - just an update. After applying this update, the State would then be {"input1": 2, "input2": ["hi"]}. If the second node returns {"input2": ["bye"]} then the State would then be {"input1": 2, "input2": ["bye"]}

#Example with the reducer attached

class State(TypedDict):
    input1:str
    # here no reducer is attached , so values will be appended
    input2:Annotated[list[str],add]


# {"input1": 1, "input2": ["hi"]}. Let’s then assume the first Node returns {"input1": 2}. This is treated as an update to the state. Notice that the Node does not need to return the whole State schema - just an update. After applying this update, the State would then be {"input1": 2, "input2": ["hi"]}. If the second node returns {"input2": ["bye"]} then the State would then be {"input1": 2, "input2": ["hi","bye"]}

# %%
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama

# %% 
LARGE_MODEL = "deepseek-r1:14b"

TIMEOUT = 120.0


# %% Tools
def naming_generator_from_int(x: int) -> str:
    """Generate a artificial name from an integer."""
    return f"Mustermann {x}"


naming_tool = FunctionTool.from_defaults(fn=naming_generator_from_int)


def multiply(x: float, y: float) -> float:
    """Multiply two numbers"""
    return x * y


def addition(x: float, y: float) -> float:
    """Add two numbers"""
    return x + y


def subtraction(x: float, y: float) -> float:
    """Subtract two numbers"""
    return x - y


def division(x: float, y: float) -> float:
    """Divide two numbers"""
    return x / y


math_tools = [
    FunctionTool.from_defaults(fn=multiply),
    FunctionTool.from_defaults(fn=addition),
    FunctionTool.from_defaults(fn=subtraction),
    FunctionTool.from_defaults(fn=division),
]

# %%
usr_msg = "Generate a name from an integer, if a number is 11 and what is the result of 3 + 4 * 9 -65 / 4?"

# %% Agent and model
model = Ollama(model=LARGE_MODEL,request_timeout=TIMEOUT)
agent = ReActAgent.from_tools([naming_tool] + math_tools, llm=model, verbose=True)

# %%
answer = agent.chat(usr_msg)
print("answer: ", answer)

# %%

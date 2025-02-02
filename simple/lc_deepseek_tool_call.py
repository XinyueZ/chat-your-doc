# %% Blog: https://teetracker.medium.com/ollama-workaround-deepseek-r1-tool-support-c64dbb924da1
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.tools import tool
from langchain_experimental.llms.ollama_functions import \
    OllamaFunctions as ToolSupportChatOllama
from langchain_ollama import ChatOllama 
from pydantic import BaseModel, Field
from rich.pretty import pprint as pp

# %%
llm = ToolSupportChatOllama( # not working for ChatOllama 
    model="deepseek-r1:14b",
    format="json",
)


# %% funny tools, for fun
class NamingGeneratorFromInt(BaseModel):
    """Generate a artificial name from an integer."""

    x: int = Field(description="An integer, we will use this to generate a name.")


@tool(
    "naming_generator_from_int",
    return_direct=True,
    args_schema=NamingGeneratorFromInt,
)
def naming_generator_from_int(x: int) -> str:
    """Generate a artificial name from an integer."""
    return f"Mustermann {x}"


# %%
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            ("""You are an assistant that only answers questions.""")
        ),
        HumanMessagePromptTemplate.from_template("""<query>{query}</query>"""),
    ]
)

# %%
chain = prompt | llm.bind_tools([naming_generator_from_int])
query = "Generate a name from an integer, if a number is 11."
response = chain.invoke(input={"query": query})
pp(response)

# %%
chain = prompt | llm.with_structured_output(NamingGeneratorFromInt, include_raw=True) 
query = "Generate a name from an integer, if a number is 11."
response = chain.invoke(input={"query": query})
pp(response)

# %%

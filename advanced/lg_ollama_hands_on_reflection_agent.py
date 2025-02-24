# %%
import io
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import requests
from dotenv import load_dotenv
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.tools import tool
from langchain_community.tools import (BraveSearch, DuckDuckGoSearchRun,
                                       YahooFinanceNewsTool)
from langchain_community.tools.wikidata.tool import (WikidataAPIWrapper,
                                                     WikidataQueryRun)
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_core.messages import (AIMessage, HumanMessage, SystemMessage,
                                     ToolMessage)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import Tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, MessagesState, StateGraph
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field
from rich.console import Console
from rich.markdown import Markdown
from rich.pretty import pprint as pp

load_dotenv()
ollama_host = os.getenv("OLLAMA_HOST")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")
os.environ["JINA_API_KEY"] = os.getenv("JINA_API_KEY")
os.environ["BRAVE_SEARCH_API_KEY"] = os.getenv("BRAVE_SEARCH_API_KEY")
os.environ["LLM_TEMPERATURE"] = os.getenv("LLM_TEMPERATURE", 1.0)
os.environ["LLM_TOP_P"] = os.getenv("LLM_TOP_P", 1.0)
os.environ["LLM_TOP_K"] = os.getenv("LLM_TOP_K", 30)
os.environ["LLM_TIMEOUT"] = os.getenv("LLM_TIMEOUT", 120)

# %%
VERBOSE = int(os.getenv("VERBOSE", 1)) == 1
MAX_ITERATIONS = 3
SEED = 42

# %% LLM
qa_model = ChatOllama(
    base_url=ollama_host,
    model="llama3.1:latest",
    num_gpu=1,
    seed=SEED,
    temperature=float(os.getenv("LLM_TEMPERATURE", 1.0)),
    top_p=float(os.getenv("LLM_TOP_P", 1.0)),
    top_k=int(os.getenv("LLM_TOP_K", 30)),
)

critic_model = ChatOllama(
    base_url=ollama_host,
    model="deepseek-r1:14b",
    num_gpu=1,
    seed=SEED,
    temperature=float(os.getenv("LLM_TEMPERATURE", 1.0)),
    top_p=float(os.getenv("LLM_TOP_P", 1.0)),
    top_k=int(os.getenv("LLM_TOP_K", 30)),
)

translate_model = ChatOllama(
    base_url=ollama_host,
    model="llama3.2:latest",
    num_gpu=1,
    seed=SEED,
    temperature=float(os.getenv("LLM_TEMPERATURE", 1.0)),
    top_p=float(os.getenv("LLM_TOP_P", 1.0)),
    top_k=int(os.getenv("LLM_TOP_K", 30)),
)


# %% Tools
@tool(
    "get_current_datetime",
    return_direct=True,
    parse_docstring=True,
)
def get_current_datetime() -> str:
    """Get the current datetime."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"🧭 Retrieved current time: {current_time}")
    return current_time


class TranslateSchema(BaseModel):
    """Translate the given text from the given language to the given language."""

    text: str = Field(description="The text to translate.")
    from_lang: str = Field(description="The language to translate from.")
    to_lang: str = Field(description="The language to translate to.")


@tool(
    "translate",
    return_direct=True,
    args_schema=TranslateSchema,
    parse_docstring=True,
)
def translate(text: str, from_lang: str, to_lang: str) -> str:
    """Translate the given text from the given language to the given language."""

    return translate_model.invoke(
        [
            (
                "system",
                f"""You are a helpful translator. Translate the user sentence from {from_lang} to {to_lang}. The origin text is in <user_text></user_text>. 
Return the result without any extra information including instructions or any tags or marks.""",
            ),
            ("human", f"<user_text>{text}</user_text>"),
        ]
    ).content


class OpenUrlSchema(BaseModel):
    """Input, for opening and reading a website."""

    website_url: str = Field(..., description="Mandatory website url to read")


@tool(
    "open_url",
    return_direct=True,
    args_schema=OpenUrlSchema,
    parse_docstring=True,
)
def open_url(website_url: str) -> str:
    """A tool that can be used to read the content behind a URL of a website."""
    return requests.get(
        f"https://r.jina.ai/{website_url}",
        headers={
            "Authorization": f"Bearer {os.environ['JINA_API_KEY']}",
            "X-Return-Format": "markdown",
            "X-Timeout": "120",
        },
        timeout=60,
    ).text


web_search_description = (
    "a web-search engine. "
    "useful for when you need to answer questions about current events."
    " input should be a search query."
)
tools = [
    get_current_datetime,
    translate,
    open_url,
    YahooFinanceNewsTool(),
    Tool(
        name="Web search (Google) on the Internet",
        func=GoogleSerperAPIWrapper(k=13).run,
        description=web_search_description,
    ),
    WikidataQueryRun(api_wrapper=WikidataAPIWrapper()),
    WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),
    DuckDuckGoSearchRun(description=web_search_description),
    BraveSearch.from_api_key(
        api_key=os.getenv("BRAVE_SEARCH_API_KEY"),
        search_kwargs={"count": 10},
        description=web_search_description,
    ),
]
tools_by_names = {tool.name: tool for tool in tools}
qa_model_with_tools = qa_model.bind_tools(tools)


# %% Make Agent with LangGraph
class BetterMessagesState(MessagesState):
    curr_iter: int = Field(default=0)
    abort: bool = Field(default=False)


reflection_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            (
                """You are an advanced AI assistant specializing in meta-analysis of AI-generated content. 
Your task is to provide insightful feedback on a series of AI responses and tool calls related to a user's original question. 
The goal is to improve the AI's ability to answer effectively."""
            )
        ),
        HumanMessagePromptTemplate.from_template(
            """First, review the following information:

1. Original User Question:
<user_question>
{USER_QUESTION}
</user_question>

2. The latest AI response to the user:
<last_ai_response>
{LAST_AI_RESPONSE}
</last_ai_response>

Your task is to analyze these elements and provide a detailed reflection on the AI's performance. Follow these steps:

1. Conduct a thorough analysis by addressing the following points within <detailed_analysis> tags:

   a. Summarize the user's question and the AI's response.
   b. Quote relevant parts of the AI response and user question. Number each quote and briefly analyze its relevance and effectiveness.
   c. Evaluate how well the AI addresses the user's original question.
   d. Assess the clarity, conciseness, and relevance of the response.
   e. Analyze the appropriateness and effectiveness of any tool calls. List each parameter and whether it was properly used.
   f. Identify patterns or inconsistencies in the response.
   g. List specific examples of strengths and weaknesses in the AI's response.
   h. Consider alternative approaches the AI could have taken to answer the question.
   i. Suggest areas for improvement.
   j. Count and list the distinct topics addressed by the AI. Number each topic and briefly describe it.
   k. Check for hallucination: Determine if the AI's response contains information beyond the scope of the user's question. If so, highlight these instances by quoting the relevant parts and explaining why they might be considered hallucination.

2. Based on your analysis, provide a structured reflection request, use the following format:

{FORMAT_INSTRUCTIONS}
 
Remember to be thorough in your analysis and clear in your reflection. Your input will be crucial for enhancing the AI's performance in future interactions."""
        ),
    ]
)


reflection_structured_output_parser = StructuredOutputParser.from_response_schemas(
    [
        ResponseSchema(
            name="Has Reflection",
            description="'yes' if reflection is required, 'no' otherwise",
            type="string",
        ),
        ResponseSchema(
            name="Overall Assessment",
            description="Provide a concise summary of the AI's performance",
            type="string",
        ),
        ResponseSchema(
            name="Detailed Analysis",
            description="""1. Content:
   [Evaluate the accuracy and completeness of the information provided]

2. Tone:
   [Assess the appropriateness of the AI's tone and language]

3. Structure:
   [Comment on the organization and flow of the response]

4. Strengths:
   [Highlight what the AI did well]

5. Areas for Improvement:
   [Identify specific aspects that need enhancement]

6. Tool Usage:
   [If applicable, evaluate the effectiveness of tool calls]

7. Hallucination Check:
   [Report on any instances of the AI providing information beyond the scope of the user's question]
""",
            type="List[string]",
        ),
        ResponseSchema(
            name="Conclusion and Recommendations",
            description="Summarize key observations and provide actionable suggestions for improvement",
            type="string",
        ),
    ]
)

str_output_parser = StrOutputParser()


critic_message_format = """You are an AI assistant tasked with improving your chat responses based on feedback and reflection. Your goal is to provide more accurate, helpful, and engaging answers to user original question.

First, review the original user question and the last AI response to the user:

Original User Question:

<user_question>
{USER_QUESTION}
</user_question>

The latest AI response to the user:
<last_ai_response>
{LAST_AI_RESPONSE}
</last_ai_response>

Now, consider the reflection provided by another AI on your previous responses:

<ai_reflection>
{AI_REFLECTION}
</ai_reflection>

Your task is to analyze the last AI response to the user, the AI reflection, and the new user question to formulate an improved response. Consider the following:

1. What aspects of your previous responses were praised in the AI reflection?
2. What areas for improvement were identified?
3. How can you apply these insights to better answer the new user question?
4. Are there any patterns or themes in the user's questions that you should address?
5. Should recall the tools to improve the response?

Based on this analysis, provide an improved response to the user's new question. Your response should demonstrate:

- Increased accuracy and relevance
- Better engagement with the user's specific concerns
- Application of insights from the AI reflection
- Improved clarity and coherence

Format your response as follows:

<improved_response>
[Your improved answer to the user's question]
</improved_response>

<reflection>
[A brief explanation of how you improved your response based on the AI reflection and previous conversation]
</reflection>

Remember to maintain a helpful and friendly tone throughout your response."""


def get_reflection_content(reflection: str) -> str:
    # logger.debug(f"Before struct-output reflection: {reflection}")
    structured_output: dict[str, Any] = reflection_structured_output_parser.parse(
        reflection
    )
    # logger.debug(f"Struct-output reflection: {structured_output}")
    if structured_output["Has Reflection"] == "yes":
        reflection_content = json.dumps(structured_output)
        return reflection_content
    return ""


def get_improved_response(message_content: str) -> str:
    improved_response = re.search(
        r"<improved_response>(.*?)</improved_response>",
        message_content,
        re.DOTALL,
    )
    if improved_response:
        return improved_response.group(1)
    else:
        return message_content


def qa(state: BetterMessagesState):
    response = qa_model_with_tools.invoke(state["messages"])
    logger.info(f"🤖 Responsed...")
    if VERBOSE:
        logger.debug(f"🤖 response: {response}")
    return {"messages": [response]}


def reflect(state: BetterMessagesState):
    whole_conversation = state["messages"]
    user_question = intermediate_steps = [
        msg
        for msg in whole_conversation
        if isinstance(msg, HumanMessage)
        and msg.additional_kwargs.get("internal", None) is None
    ][
        -1
    ]  # Ensure the last HumanMessage is a question not a critic.
    intermediate_steps = [
        msg for msg in whole_conversation if isinstance(msg, AIMessage)
    ]  # filter out all AIMessage
    last_ai_response_content = get_improved_response(intermediate_steps[-1].content)
    # Reflection ##################################################################################
    logger.info(f"💥 reflection started ({state['curr_iter']})")
    reflection_chain = reflection_prompt | critic_model | str_output_parser
    reflection = reflection_chain.invoke(
        input={
            "USER_QUESTION": user_question.content,
            "LAST_AI_RESPONSE": last_ai_response_content,
            "FORMAT_INSTRUCTIONS": reflection_structured_output_parser.get_format_instructions(),
        }
    )
    reflection_content = get_reflection_content(reflection)
    if VERBOSE:
        logger.debug(f"💡 reflection ended: {reflection}")
        logger.debug(f"💡💡 reflection content: {reflection_content}")
    if len(reflection_content.strip()) == 0:
        logger.info(f"👍 Critic free")
        return {
            "abort": True,
            "curr_iter": state["curr_iter"] + 1,
        }
    # Reflection done #############################################################################

    # Tell QA-model to improve answer #############################################################
    critic_message = critic_message_format.format(
        USER_QUESTION=user_question.content,
        LAST_AI_RESPONSE=last_ai_response_content,
        AI_REFLECTION=reflection_content,
    )
    if VERBOSE:
        logger.debug(f"👮‍♂ critic: {critic_message}")
    logger.info(f"👍 Critic pass-To-AI")
    return {
        "messages": [
            HumanMessage(
                content=critic_message, additional_kwargs={"internal": "critic"}
            )
        ],
        "abort": False,
        "curr_iter": state["curr_iter"] + 1,
    }


def post_qa(state: BetterMessagesState) -> Literal["use_tools", "reflect", END]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "use_tools"
    if state["curr_iter"] > MAX_ITERATIONS - 1:
        return END
    return "reflect"


def post_reflect(state: BetterMessagesState) -> Literal["qa", END]:
    if state["abort"]:
        return END
    return "qa"


def use_tools(state: dict):
    tool_results = []
    for tool_call in state["messages"][-1].tool_calls:
        try:
            observation = tools_by_names[tool_call["name"]].invoke(
                input=tool_call["args"]
            )
            logger.info(f"🔧 Tool used: {tool_call['name']}")
        except Exception as e:
            observation = f"Error performing search: {str(e)} on {tool_call['name']}, try an alternative tool."
            logger.error(f"🔧 Tool failed: {tool_call['name']}")
            if VERBOSE:
                logger.error(f"🔧 Tool failed: {e}")
        tool_message = ToolMessage(content=observation, tool_call_id=tool_call["id"])
        tool_results.append(tool_message)
    if VERBOSE:
        logger.debug(f"🔧 tool_results: {tool_results}")
    return {"messages": tool_results}


agent_builder = StateGraph(BetterMessagesState)
agent_builder.add_node("qa", qa)
agent_builder.add_node("use_tools", use_tools)
agent_builder.add_node("reflect", reflect)

agent_builder.add_edge(START, "qa")
agent_builder.add_conditional_edges(
    "qa",
    post_qa,
    {
        "use_tools": "use_tools",
        "reflect": "reflect",
        END: END,
    },
)
agent_builder.add_edge("use_tools", "qa")
agent_builder.add_conditional_edges(
    "reflect",
    post_reflect,
    {
        "qa": "qa",
        END: END,
    },
)


agent = agent_builder.compile()


# %%
if __name__ == "__main__":
    # %% User query
    messages = [
        SystemMessage(
            "You are a helpful assistant tasked with generating artificial names from integers."
            "You can use different tools to help you to complete any task."
        ),
        HumanMessage(
            content="""I want to buy a MacBook Pro Max (M4), here is the link to the product: https://www.apple.com/de/shop/buy-mac/macbook-pro/14-zoll-m4-max
Can you use the price trends of the MacBook Pro Max (M4) over the past three years and the expected price trends for this year to provide a suitable time for me to purchase this model in 2025? 

IMPORTANT: 
- Use the Internet(mandatory), Wikipedia or WikiData and so on as the most reliable source.
- Only MacBook Pro Max (M4) is our goal, don't consider other types or models.
- Result in Markdown format.
- Currency in Euro (€), you can use the actual rate from the internet.
- Location in Germany, you can scale to worldwide.

Give me the answer as follows:
0. Explain our goals, which is also the main purpose of this report.
1. Purchase timing, seasonal adjustments
2. Expected price
3. Minimum configuration for the MacBook Pro Max (M4) might be
4. Conclusion and Recommendations: Summarize key observations and provide actionable suggestions.

Begin to analyze and generate an answer:
"""
        ),
    ]

    # TODO: Wrapper into a function to save graph image
    img_bytes = agent.get_graph().draw_mermaid_png()
    image = Image.open(io.BytesIO(img_bytes))
    graph_image_output_local = os.path.join(
        "./workflow", f"lg_ollama_hands_on_reflection_agent.png"
    )
    Path(graph_image_output_local).parent.mkdir(parents=True, exist_ok=True)
    image.save(graph_image_output_local)
    # Saved graph image

    response = agent.invoke(
        {"messages": messages, "abort": False, "curr_iter": 0},
        {"recursion_limit": 9999},
    )
    # %%
    if VERBOSE:
        pp(response)
        for m in response["messages"]:
            m.pretty_print()
    # %% Extract response
    last_content = response["messages"][-1].content
    # %% Display response
    improved_response = get_improved_response(last_content)
    console = Console(record=True, soft_wrap=True)
    md = Markdown(improved_response, justify="left")
    console.print(md)
    file_to_save = "output/lg_ollama_hands_on_reflection_agent_improved_response.md"
    console.save_text(file_to_save, styles=False)
    logger.success(f"Saved to {file_to_save}")
    # %%

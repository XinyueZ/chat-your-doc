import asyncio
import base64
import io
import os
import sys
import uuid
from datetime import datetime
from functools import partial
from inspect import getframeinfo, stack
from typing import Any, Dict, List, Tuple

import nest_asyncio
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance
from langchain import hub
from langchain.agents import Tool, load_tools
from langchain.agents.agent import AgentExecutor
from langchain.agents.structured_chat.base import create_structured_chat_agent
from langchain.memory import ChatMessageHistory
from langchain.prompts import HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.wikidata.tool import (WikidataAPIWrapper,
                                                     WikidataQueryRun)
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.vectorstores import FAISS
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.tool import ToolMessage, ToolMessageChunk
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from matplotlib import pyplot as plt
from PIL import Image, UnidentifiedImageError
from rich.pretty import pprint
from ultralytics import YOLO
from ultralytics.engine.results import Boxes, Results

VERBOSE = True
MAX_TOKEN = 2048

OPENAI_LLM = "gpt-4o"
GOOGLE_LLM = "gemini-1.5-flash-latest"
CV_MODEL: str = "yolov8s-world.pt"

FUN_MAPPING = {}

nest_asyncio.apply()

st.set_page_config(layout="wide")
# ------------------------------------------helpers------------------------------------------


def pretty_print(title: str = "Untitled", content: Any = None):
    if not VERBOSE:
        return

    info = getframeinfo(stack()[1][0])
    print()
    pprint(
        f":--> {title} --> {info.filename} --> {info.function} --> line: {info.lineno} --:"
    )
    pprint(content)


def image_url_to_image(image_url: str) -> Image:
    return Image.open(requests.get(image_url, stream=True).raw)


def create_random_filename(ext=".txt") -> str:
    return create_random_name() + ext


def create_random_name() -> str:
    return str(uuid.uuid4())


def doc_uploader() -> bytes:
    with st.sidebar:
        uploaded_doc = st.file_uploader("# Upload one image", key="doc_uploader")
        if not uploaded_doc:
            st.session_state["file_name"] = None
            st.session_state["base64_image"] = None
            # pretty_print("doc_uploader", "No image uploaded")
            return None
        if uploaded_doc:
            tmp_dir = "./chat-your-doc/tmp/"
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir)
            temp_file_path = os.path.join(tmp_dir, f"{uploaded_doc.name}")
            with open(temp_file_path, "wb") as file:
                file.write(uploaded_doc.getvalue())
                file_name = uploaded_doc.name
                # pretty_print("doc_uploader", f"Uploaded {file_name}")
                uploaded_doc.flush()
                uploaded_doc.close()
                # os.remove(temp_file_path)
                if st.session_state.get("file_name") == file_name:
                    # pretty_print("doc_uploader", "Same file")
                    return st.session_state["base64_image"]

                # pretty_print("doc_uploader", "New file")
                st.session_state["file_name"] = temp_file_path
                with open(temp_file_path, "rb") as image_file:
                    st.session_state["base64_image"] = base64.b64encode(
                        image_file.read()
                    ).decode("utf-8")

                return st.session_state["base64_image"]
        return None


# ------------------------------------------LLM Chain------------------------------------------


def create_message_chain(model: BaseChatModel, base64_image: bytes) -> Runnable:
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=[
                    {
                        "type": "text",
                        "text": """As a helpful assistant, you should respond to the user's query.
Avoid giving any information related to the local file system or sandbox.""",
                    }
                ]
            ),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template(
                template=(
                    [
                        {
                            "type": "text",
                            "text": "{query}",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ]
                    if base64_image
                    else [
                        {
                            "type": "text",
                            "text": "{query}",
                        },
                    ]
                )
            ),
        ]
    )
    return prompt | model


def create_tool_chain(model: BaseChatModel) -> Runnable:
    return model


# ------------------------------------------agent, functions, tools------------------------------------------
class LoadUrlsTool(BaseModel):
    query: str = Field(description="The query to search for.")
    urls: List[str] = Field(description="The URLs to load.")


@tool("load-urls-tool", args_schema=LoadUrlsTool, return_direct=False)
def load_urls_tool(retrieval_model: BaseChatModel, query: str, urls: List[str]) -> str:
    """Load the content of the given Urls for getting responses to the query and return the query result."""

    load_urls_prompt_template = PromptTemplate.from_template(
        """Reponse the query inside [query] based on the context inside [context]:
[query]
{query}
[query]

[context]
{context}
[context]

Only return the answer without any instruction text or additional information.
Keep the result as simple as possible."""
    )
    loader = WebBaseLoader(urls)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0
    )
    chunks = splitter.split_documents(docs)
    db = FAISS.from_documents(chunks, OpenAIEmbeddings())
    retriever = db.as_retriever()

    chain = (
        {"context": retriever, "query": RunnablePassthrough()}
        | load_urls_prompt_template
        | retrieval_model
        | StrOutputParser()
    )
    return chain.invoke(query)


search_agent_tools = [
    Tool(
        name="Google Search",
        description="Search Google for recent results.",
        func=GoogleSearchAPIWrapper(
            google_api_key=os.environ.get("GOOGLE_CSE_KEY")
        ).run,
    ),
    Tool(
        name="DuckDuckGo Search",
        func=DuckDuckGoSearchRun().run,
        description="Use to search for the information from DuckDuckGo.",
    ),
    Tool(
        name="Wikipedia Search",
        func=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()).run,
        description="Use to search for the information from Wikipedia.",
    ),
    Tool(
        name="Wikidata Search",
        func=WikidataQueryRun(api_wrapper=WikidataAPIWrapper()).run,
        description="Use to search for the information from Wikidata.",
    ),
]
search_agent_tools.extend(load_tools(["arxiv"]))


def create_search_agent(agent_model: BaseChatModel) -> AgentExecutor:
    return AgentExecutor(
        agent=create_structured_chat_agent(
            llm=agent_model,
            tools=search_agent_tools,
            prompt=hub.pull("hwchase17/structured-chat-agent"),
        ),
        tools=search_agent_tools,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )


# ------------------------------------------tool functions------------------------------------------


def generate_image(model: BaseChatModel, context: str) -> str:
    """Generate an image to illustrate for user request."""

    tools = load_tools(["dalle-image-generator"])
    agent = create_structured_chat_agent(
        llm=model,
        tools=tools,
        prompt=hub.pull("hwchase17/structured-chat-agent"),
    )
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )
    prompt = f"""Generate an image to illustrate for user request with the following context:

Context:
{context}

Notice: 
- ONLY return the image link.
- WHEN the image includes text, it MUST be in the same language as the language of the input text.
"""
    image_gen = agent_executor.invoke({"input": prompt})
    # pretty_print("Image Gen:", image_gen)
    try:
        image_url = image_gen["output"]
        return image_url_to_image(image_url), image_url
    except UnidentifiedImageError:
        return None, None


def annotate_image(
    model: BaseChatModel,
    base64_image: bytes,
    image_description: str,
) -> Tuple[Results, str]:
    """Annotate an image with classes (COCO dataset types), response results with bounding boxes."""

    def coco_label_extractor(model: BaseChatModel, image_description: str) -> str:
        """Read an image description and extract COCO defined labels as much as possible from the description."""
        template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You as an AI assistant can understand an image descritpion. 
                 Try to extract COCO defined labels as much as possible from the description.
                 Only return lables and split by comma, no empty space.""",
                ),
                ("human", "Image descritpion: {img_desc}"),
            ]
        )
        human_input = template.format_messages(img_desc=image_description)
        return model.invoke(human_input).content

    # pretty_print("Image description:", image_description)
    classes = coco_label_extractor(model, image_description)
    # pretty_print("Classes:", classes)
    classes = classes.split(",") if classes else list()
    model = YOLO(CV_MODEL)  # or select yolov8m/l-world.pt for different sizes

    if classes is not None and len(classes) > 0:
        model.set_classes(classes)

    image = Image.open(io.BytesIO(base64.b64decode(base64_image)))
    preds = model.predict(image)
    results: Results = preds[0]
    save_dir = "tmp"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = f"{save_dir}/annotated_{create_random_filename('.jpg')}"
    results.save(save_path)
    return results, save_path


def run_search_agent(agent: AgentExecutor, topic: str) -> Dict[str, Any]:
    """Run the agent for the topic, the agent will search the topic through the web and return the result."""
    prompt = f"""Search through the internet for the topic inside >>>>>>>>> and <<<<<<<<<<
>>>>>>>>>
{topic}
<<<<<<<<<<
We recommend certain tools that you can use:
-  Google Search
-  DuckDuckGo Search
-  Wikipedia Search
-  Wikidata Search
-  arxiv Search
Also if you face some urls for more details you can also use the tool:
- load-urls-tool
"""
    return agent.invoke({"input": prompt})


def get_current_time(_: str) -> str:
    """For questions or queries that are relevant to current time or date"""

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return current_time


def get_and_plot_stock_prices(
    stock_symbols: List[str], start_date: str, end_date: str
) -> Tuple[pd.DataFrame, str]:
    """Get and plot the stock prices for the given stock symbols between
    the start and end dates.

    Args:
        stock_symbols (str or list): The stock symbols to get the
        prices for.
        start_date (str): The start date in the format
        'YYYY-MM-DD'.
        end_date (str): The end date in the format 'YYYY-MM-DD'.

    Returns:
            Tuple:
                pandas.DataFrame: The stock prices for the given stock symbols indexed by date, with one column per stock symbol.
                stock_prices (pandas.DataFrame): The stock prices for the
                given stock symbols.
    """

    def _plot_stock_prices(stock_prices: pd.DataFrame, filename: str):
        """Plot the stock prices for the given stock symbols.
        Args:
            stock_prices (pandas.DataFrame): The stock prices for the given stock symbols.
            filename (str): The filename to save the plot to.

        """
        plt.figure(figsize=(10, 5))
        for column in stock_prices.columns:
            plt.plot(stock_prices.index, stock_prices[column], label=column)
        plt.title("Stock Prices")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend(stock_prices)
        plt.grid(True)
        plt.savefig(filename)

    stock_data = yfinance.download(stock_symbols, start=start_date, end=end_date)
    pretty_print("stock_data:", stock_data)
    prices_df = stock_data.get("Close")
    pretty_print("Close:", prices_df)

    plot_filename = create_random_filename(ext=".png")
    fullpath = f"./tmp/{plot_filename}"

    _plot_stock_prices(prices_df, fullpath)
    return (prices_df, fullpath)


# ------------------------------------------tool------------------------------------------
class GenerateImageTool(BaseModel):
    """Generate an image to illustrate for user request."""

    context: str = Field(
        ...,
        description="The context for generating an image to illustrate what the user requested.",
    )


class AnnotateImageTool(BaseModel):
    """Annotate an image with classes (COCO dataset types), response results with bounding boxes."""

    image_description: str = Field(
        ...,
        description="The description of the image that needs to be annotated.",
    )


class RunSearchAgentTool(BaseModel):
    """Run the agent to search for corresponding topic. The agent will search the web for the topic and return the results."""

    topic: str = Field(
        ...,
        description="The topic to search through the web.",
    )


class GetCurrentTimeTool(BaseModel):
    """Get the current time."""

    event: str = Field(
        ...,
        description="Some questions or queries that are relevant to current time or date",
    )


class GetAndPlotStockPrices(BaseModel):
    """Get and plot the stock prices for the given stock symbols between the start and end dates."""

    stock_symbols: str = Field(
        ...,
        description="The stock symbols list string to get prices for, should be a string with comma-separated different symbols.",
    )
    start_date: str = Field(
        ...,
        description="The start date in the format 'YYYY-MM-DD' or other understandable format.",
    )
    end_date: str = Field(
        ...,
        description="The end date in the format 'YYYY-MM-DD' or other understandable format.",
    )


# ------------------------------------------model event handlers------------------------------------------
def handle_generate_image(
    tool_name: str,
    tool_id: str,
    context: str,
) -> ToolMessage:
    additional_kwargs = dict()
    try:
        func = FUN_MAPPING.get(tool_name, None)
        if func:
            image, image_url = func(context=context)
        else:
            raise ValueError("No tool provided.")

        additional_kwargs = {"image_url": image_url} if image_url else {}
        return ToolMessage(
            content=(
                f"Generated image successfully, tool has finished: ![]({image_url})"
                if image_url
                else "Tool called, nothing was generated"
            ),
            tool_call_id=tool_id,
            additional_kwargs=additional_kwargs,
        )

    except Exception as e:
        st.write(f"Something went wrong.\n\n{e}")
        return ToolMessage(
            content=f"Tool was called but failed to generate image.\n\n{e}",
            tool_call_id=tool_id,
            additional_kwargs=additional_kwargs,
        )


def handle_annotate_image(
    tool_name: str,
    tool_id: str,
    base64_image: bytes,
    image_description: str,
) -> ToolMessage:
    additional_kwargs = {}
    try:
        func = FUN_MAPPING.get(tool_name, None)
        if func:
            _, image_path = func(
                base64_image=base64_image, image_description=image_description
            )
        else:
            raise ValueError("No tool provided.")

        additional_kwargs["image_path"] = image_path if image_path else ""
        return ToolMessage(
            content=(
                f"Annotated image successfully, tool has finished."
                if image_path
                else "Tool called, nothing was annotated"
            ),
            tool_call_id=tool_id,
            additional_kwargs=additional_kwargs,
        )

    except Exception as e:
        st.write(f"Something went wrong.\n\n{e}")
        return ToolMessage(
            content=f"Tool was called but failed to annotate image.\n\n{e}",
            tool_call_id=tool_id,
            additional_kwargs=additional_kwargs,
        )


def handle_search_agent(
    tool_name: str,
    tool_id: str,
    topic: str,
) -> ToolMessage:
    additional_kwargs = {}
    try:
        func = FUN_MAPPING.get(tool_name, None)
        if func:
            agent_res = func(topic)
        else:
            raise ValueError("No tool provided.")

        additional_kwargs["string"] = agent_res["output"]
        return ToolMessage(
            content=(
                f"Searched successfully, tool has finished:\n\n{agent_res['output']}"
                if agent_res["output"] and agent_res["output"] != ""
                else "Tool called, nothing was responsed by agent."
            ),
            tool_call_id=tool_id,
            additional_kwargs=additional_kwargs,
        )

    except Exception as e:
        st.write(f"Something went wrong.\n\n{e}")
        return ToolMessage(
            content=f"Tool was called but failed to get result.\n\n{e}",
            tool_call_id=tool_id,
            additional_kwargs=additional_kwargs,
        )


def handle_get_current_time(
    tool_name: str,
    tool_id: str,
) -> ToolMessage:
    additional_kwargs = {}
    try:
        func = FUN_MAPPING.get(tool_name, None)
        if func:
            current_time = func("")
        else:
            raise ValueError("No tool provided.")

        additional_kwargs["string"] = f"Current time: {current_time}"
        return ToolMessage(
            content=(
                f"Get current time: {current_time}, tool has finished."
                if current_time and current_time != ""
                else "Tool called, nothing was responsed by agent."
            ),
            tool_call_id=tool_id,
            additional_kwargs=additional_kwargs,
        )

    except Exception as e:
        st.write(f"Something went wrong.\n\n{e}")
        return ToolMessage(
            content=f"Tool was called but failed to get current time.\n\n{e}",
            tool_call_id=tool_id,
            additional_kwargs=additional_kwargs,
        )


def handle_get_stock_prices(
    tool_name: str,
    tool_id: str,
    tool_stock_symbols: str,
    tool_start_date: str,
    tool_end_date: str,
) -> ToolMessage:
    additional_kwargs = {}
    try:
        func = FUN_MAPPING.get("GetAndPlotStockPrices", None)
        if func:
            prices_df, image_path = func(
                stock_symbols=tool_stock_symbols.split(","),
                start_date=tool_start_date,
                end_date=tool_end_date,
            )
        else:
            raise ValueError("No tool provided.")

        additional_kwargs["image_path"], additional_kwargs["dataframe"] = (
            image_path,
            prices_df,
        )

        return ToolMessage(
            content=f"""Complete getting stock prices successfully, tool has finished, stop further investigation. The stock prices requested:
dataframe:
{prices_df}""",
            tool_call_id=tool_id,
            additional_kwargs=additional_kwargs,
        )

    except Exception as e:
        st.write(f"Something went wrong.\n\n{e}")
        return ToolMessage(
            content=f"Tool was called but failed to get stock prices completely: {e}",
            tool_call_id=tool_id,
            additional_kwargs=additional_kwargs,
        )


# ------------------------------------------chat-----------------------------------------------
def tool_call_proc(
    tool_call: Dict,
    base64_image: bytes = None,
) -> ToolMessage:
    tool_id, tool_name = tool_call["id"], tool_call["name"]
    match tool_name:
        case "GenerateImageTool":
            context = tool_call["args"]["context"]
            return handle_generate_image(tool_name, tool_id, context)
        case "AnnotateImageTool":
            image_description = tool_call["args"]["image_description"]
            return handle_annotate_image(
                tool_name, tool_id, base64_image, image_description
            )
        case "RunSearchAgentTool":
            topic = tool_call["args"]["topic"]
            return handle_search_agent(tool_name, tool_id, topic)
        case "GetCurrentTimeTool":
            return handle_get_current_time(tool_name, tool_id)
        case "GetAndPlotStockPrices":
            now = datetime.now()
            tool_stock_symbols = tool_call["args"]["stock_symbols"]
            tool_start_date = tool_call["args"].get("start_date", now)
            tool_end_date = tool_call["args"].get("end_date", now)
            return handle_get_stock_prices(
                tool_name, tool_id, tool_stock_symbols, tool_start_date, tool_end_date
            )
        case _:
            return ToolMessage(
                content=f"Handle the UNKNOWN tool call: {tool_name}",
                tool_call_id=tool_id,
                additional_kwargs=dict(),
            )


def chat_with_model(
    model: BaseChatModel,
    base64_image: bytes = None,
    streaming=False,
    image_width=500,
):
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history" not in st.session_state:
        st.session_state.history = ChatMessageHistory()
    for message in st.session_state.messages:
        # pretty_print("message", message)
        if message["role"] == "tool":
            if message["additional_kwargs"].get("image_path"):
                with st.chat_message("assistant"):
                    st.image(
                        message["additional_kwargs"]["image_path"], width=image_width
                    )
        else:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if prompt := st.chat_input("Write...", key="chat_input"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                should_continue, tool_messages = True, None
                while should_continue:
                    if not tool_messages:
                        chat_chain = RunnableWithMessageHistory(
                            create_message_chain(
                                model,
                                base64_image,
                            ),
                            lambda _: st.session_state.history,
                            input_messages_key="query",
                            history_messages_key="history",
                        )
                        call_model = (
                            chat_chain.invoke if not streaming else chat_chain.stream
                        )
                        res = call_model(
                            {"query": prompt},
                            {"configurable": {"session_id": None}},
                        )
                    else:
                        chat_chain = RunnableWithMessageHistory(
                            create_tool_chain(model), lambda _: st.session_state.history
                        )
                        call_model = (
                            chat_chain.invoke if not streaming else chat_chain.stream
                        )
                        res = call_model(
                            {None: tool_messages},
                            {"configurable": {"session_id": None}},
                        )
                        st.session_state.history.messages = (
                            st.session_state.history.messages[:-1]
                            + tool_messages
                            + st.session_state.history.messages[-1:]
                        )
                        del tool_messages

                    content, additional_kwargs, tool_calls = None, dict(), None
                    if not streaming:
                        content = res.content
                        st.markdown(content)
                    else:
                        content = st.write_stream(res)

                    pretty_print("history", st.session_state.history)
                    last_ai_msg = st.session_state.history.messages[-1]

                    def _has_tool_calls() -> bool:
                        return len(last_ai_msg.tool_calls) > 0

                    async def _run_tool_call_proc(tool_call: Dict) -> ToolMessage:
                        tool_msg = tool_call_proc(tool_call, base64_image)
                        additional_kwargs = tool_msg.additional_kwargs
                        if additional_kwargs.get("image_path"):
                            st.image(
                                additional_kwargs.get("image_path"),
                                width=image_width,
                            )
                        return tool_msg

                    if _has_tool_calls():
                        streaming, should_continue = False, True
                        tool_calls = last_ai_msg.tool_calls
                        tasks = [
                            _run_tool_call_proc(tool_call) for tool_call in tool_calls
                        ]
                        tool_messages = asyncio.run(asyncio.gather(*tasks))
                    else:
                        should_continue = False

                    if not _has_tool_calls():
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": content,
                                "additional_kwargs": additional_kwargs,
                            }
                        )
                    else:
                        for tool_msg in tool_messages:
                            st.session_state.messages.append(
                                {
                                    "role": "tool",
                                    "content": tool_msg.content,
                                    "additional_kwargs": tool_msg.additional_kwargs,
                                }
                            )


# ------------------------------------------main, app entry------------------------------------------
async def main():
    base64_image = doc_uploader()
    if base64_image:
        st.sidebar.image(st.session_state["file_name"], use_column_width=True)
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0, key="key_temperature")
    model_sel = st.sidebar.selectbox("Model", ["GPT-4o", "Gemini Pro"], index=0)
    if model_sel == "Gemini Pro":
        used_model = ChatGoogleGenerativeAI(
            model=GOOGLE_LLM,
            temperature=st.session_state.key_temperature,
            max_tokens=MAX_TOKEN,
        )
    else:
        used_model = ChatOpenAI(
            model=OPENAI_LLM,
            temperature=st.session_state.key_temperature,
            max_tokens=MAX_TOKEN,
        )
    ######################## config tool-bindings ########################
    partial_generate_image = partial(
        generate_image,
        model=ChatOpenAI(
            model=OPENAI_LLM,
            temperature=st.session_state.key_temperature,
            max_tokens=MAX_TOKEN,
        ),
    )
    partial_annotate_image = partial(
        annotate_image,
        used_model,
    )
    partial_run_search_agent = partial(
        run_search_agent, create_search_agent(used_model)
    )
    search_agent_tools.extend([partial(load_urls_tool, used_model)])

    FUN_MAPPING["GenerateImageTool"] = partial_generate_image
    FUN_MAPPING["AnnotateImageTool"] = partial_annotate_image
    FUN_MAPPING["RunSearchAgentTool"] = partial_run_search_agent
    FUN_MAPPING["GetCurrentTimeTool"] = get_current_time
    FUN_MAPPING["GetAndPlotStockPrices"] = get_and_plot_stock_prices

    used_model = used_model.bind_tools(
        [
            GenerateImageTool,
            AnnotateImageTool,
            RunSearchAgentTool,
            GetCurrentTimeTool,
            GetAndPlotStockPrices,
        ]
    )
    ########################################################################
    chat_with_model(
        used_model,
        base64_image,
        streaming=st.session_state.get("key_streaming", True),
        image_width=st.session_state.get("key_width", 300),
    )
    streaming = st.sidebar.checkbox("Streamming", True, key="key_streaming")
    image_width = st.sidebar.slider("Image Width", 100, 1000, 500, 100, key="key_width")
    if not os.environ.get("GOOGLE_CSE_ID") or not os.environ.get("GOOGLE_CSE_KEY"):
        st.warning(
            """For Google Search, set GOOGLE_CSE_ID, details: 
key: https://developers.google.com/custom-search/docs/paid_element#api_key
; what: https://support.google.com/programmable-search/answer/12499034?hl=en
; enable: https://console.cloud.google.com/apis/library/customsearch.googleapis
"""
        )


if __name__ == "__main__":
    asyncio.run(main())

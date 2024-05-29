import asyncio
import base64
import io
import os
import sys
import uuid
from functools import partial
from inspect import getframeinfo, stack
from typing import Any, Dict, List, Tuple

import nest_asyncio
import numpy as np
import requests
import streamlit as st
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
            pretty_print("doc_uploader", "No image uploaded")
            return None
        if uploaded_doc:
            tmp_dir = "./chat-your-doc/tmp/"
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir)
            temp_file_path = os.path.join(tmp_dir, f"{uploaded_doc.name}")
            with open(temp_file_path, "wb") as file:
                file.write(uploaded_doc.getvalue())
                file_name = uploaded_doc.name
                pretty_print("doc_uploader", f"Uploaded {file_name}")
                uploaded_doc.flush()
                uploaded_doc.close()
                # os.remove(temp_file_path)
                if st.session_state.get("file_name") == file_name:
                    pretty_print("doc_uploader", "Same file")
                    return st.session_state["base64_image"]

                pretty_print("doc_uploader", "New file")
                st.session_state["file_name"] = temp_file_path
                with open(temp_file_path, "rb") as image_file:
                    st.session_state["base64_image"] = base64.b64encode(
                        image_file.read()
                    ).decode("utf-8")

                return st.session_state["base64_image"]
        return None


# ------------------------------------------LLM Chain------------------------------------------


def create_chain(model: BaseChatModel, base64_image: bytes) -> Runnable:
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=[
                    {
                        "type": "text",
                        "text": """As a helpful assistant, you should respond to the user's query.""",
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


# ------------------------------------------agent, functions, tools------------------------------------------
class LoadUrlsTool(BaseModel):
    query: str = Field(description="The query to search for.")
    urls: List[str] = Field(description="The URLs to load.")


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


@tool("load-urls-tool", args_schema=LoadUrlsTool, return_direct=False)
def load_urls_tool(retrieval_model: BaseChatModel, query: str, urls: List[str]) -> str:
    """Load the content of the given Urls and return the query result."""
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


agent_tools = [
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


def create_agent(agent_model: BaseChatModel) -> AgentExecutor:
    return AgentExecutor(
        agent=create_structured_chat_agent(
            llm=agent_model,
            tools=agent_tools,
            prompt=hub.pull("hwchase17/structured-chat-agent"),
        ),
        tools=agent_tools,
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
    pretty_print("Image Gen:", image_gen)
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

    pretty_print("Image description:", image_description)
    classes = coco_label_extractor(model, image_description)
    pretty_print("Classes:", classes)
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


def run_agent(agent: AgentExecutor, known_topic: str) -> Dict[str, Any]:
    """Run the agent for the unknown topic, the agent will search the topic through the web and return the result."""
    prompt = f"""Search through the internet for the unknown topic inside >>>>>>>>> and <<<<<<<<<<
>>>>>>>>>
{known_topic}
<<<<<<<<<<
We recommend certain tools that you can use:
-  Google Search
-  DuckDuckGo Search
-  Wikipedia Search
-  Wikidata Search
also if you face some urls you can also use the tool:
- load-urls-tool
"""
    return agent.invoke({"input": prompt})


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


class RunAgentTool(BaseModel):
    """Run the agent for the unknown topic, an agent will search the topic through the web and return the result."""

    known_topic: str = Field(
        ...,
        description="The known topic to search for the unknown topic, an agent will use this topic to search through the web.",
    )


# ------------------------------------------model event handlers------------------------------------------
def handle_generate_image(
    tool_name: str,
    tool_id: str,
    history_messages: List[BaseMessage],
    context: str,
    image_width=500,
) -> Dict[str, str]:
    additional_kwargs = dict()
    try:
        func = FUN_MAPPING.get(tool_name, None)
        if func:
            image, image_url = func(context=context)
        else:
            raise ValueError("No tool provided.")

        if image and image_url:
            st.image(image, width=image_width)
        else:
            st.write("No image generated.")

        additional_kwargs = {"image_url": image_url} if image_url else {}

        history_messages.append(
            ToolMessage(
                content=(
                    f"Tool called and get image successfully."
                    if image_url
                    else "Tool called, nothing was generated"
                ),
                tool_call_id=tool_id,
                additional_kwargs=additional_kwargs,
            )
        )
    except Exception as e:
        st.write(f"Something went wrong.\n\n{e}")

        history_messages.append(
            ToolMessage(
                content=f"Tool was called but failed to generate image.\n\n{e}",
                tool_call_id=tool_id,
            )
        )
    return additional_kwargs


def handle_annotate_image(
    tool_name: str,
    tool_id: str,
    history_messages: List[BaseMessage],
    base64_image: bytes,
    image_description: str,
    image_width=500,
) -> Dict[str, str]:
    additional_kwargs = {}
    try:
        func = FUN_MAPPING.get(tool_name, None)
        if func:
            _, image_path = func(
                base64_image=base64_image, image_description=image_description
            )
        else:
            raise ValueError("No tool provided.")

        if image_path:
            st.image(image_path, width=image_width)
            additional_kwargs["image_path"] = image_path
        else:
            st.write("No image annotated.")

        history_messages.append(
            ToolMessage(
                content=(
                    f"Tool called and annotated the image successfully."
                    if image_path
                    else "Tool called, nothing was annotated"
                ),
                tool_call_id=tool_id,
                additional_kwargs=additional_kwargs,
            )
        )
    except Exception as e:
        st.write(f"Something went wrong.\n\n{e}")
        history_messages.append(
            ToolMessage(
                content=f"Tool was called but failed to annotate image.\n\n{e}",
                tool_call_id=tool_id,
            )
        )
    return additional_kwargs


def handle_agent_for_unknown_topic(
    tool_name: str,
    tool_id: str,
    history_messages: List[BaseMessage],
    known_topic: str,
):
    additional_kwargs = {}
    try:
        func = FUN_MAPPING.get(tool_name, None)
        if func:
            agent_res = func(known_topic)
        else:
            raise ValueError("No tool provided.")

        st.write(agent_res["output"])
        additional_kwargs["string"] = agent_res["output"]

        history_messages.append(
            ToolMessage(
                content=(
                    f"Tool called and get result successfully."
                    if agent_res["output"] and agent_res["output"] != ""
                    else "Tool called, nothing was responsed by agent."
                ),
                tool_call_id=tool_id,
                additional_kwargs={"string": agent_res["output"]},
            )
        )
    except Exception as e:
        st.write(f"Something went wrong.\n\n{e}")
        history_messages.append(
            ToolMessage(
                content=f"Tool was called but failed to get result.\n\n{e}",
                tool_call_id=tool_id,
            )
        )
    return additional_kwargs


# ------------------------------------------chat------------------------------------------
def tool_call_proc(
    tool_call: Dict,
    history_messages: List[BaseMessage],
    base64_image: bytes = None,
    image_width=500,
) -> Dict:
    tool_id, tool_name = tool_call["id"], tool_call["name"]
    match tool_name:
        case "GenerateImageTool":
            context = tool_call["args"]["context"]
            additional_kwargs = handle_generate_image(
                tool_name,
                tool_id,
                history_messages,
                context,
                image_width,
            )
            return additional_kwargs
        case "AnnotateImageTool":
            image_description = tool_call["args"]["image_description"]
            additional_kwargs = handle_annotate_image(
                tool_name,
                tool_id,
                history_messages,
                base64_image,
                image_description,
                image_width,
            )
            return additional_kwargs
        case "RunAgentTool":
            known_topic = tool_call["args"]["known_topic"]
            additional_kwargs = handle_agent_for_unknown_topic(
                tool_name,
                tool_id,
                history_messages,
                known_topic,
            )
            return additional_kwargs
        case _:
            return {}


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
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Keep image rendering while UI is being refreshed.
            additional_kwargs = message.get("additional_kwargs", None)
            if additional_kwargs and "image_url" in additional_kwargs:
                st.image(
                    image_url_to_image(additional_kwargs["image_url"]),
                    width=image_width,
                )
            elif additional_kwargs and "image_path" in additional_kwargs:
                st.image(
                    additional_kwargs["image_path"],
                    width=image_width,
                )
            elif additional_kwargs and "string" in additional_kwargs:
                st.write(additional_kwargs["string"])

    if prompt := st.chat_input("Write...", key="chat_input"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chat_chain = RunnableWithMessageHistory(
                    create_chain(
                        model,
                        base64_image,
                    ),
                    lambda _: st.session_state.history,
                    input_messages_key="query",
                    history_messages_key="history",
                )
                out_met = chat_chain.invoke if not streaming else chat_chain.stream
                res = out_met(
                    {"query": prompt},
                    {"configurable": {"session_id": None}},
                )
                content, additional_kwargs = None, dict()
                if not streaming:
                    content = res.content
                    st.write(content)
                else:
                    content = st.write_stream(res)
                pretty_print("history", st.session_state.history)
                if (
                    len(st.session_state.history.messages) > 1
                    and len(st.session_state.history.messages[-1].tool_calls) > 0
                ):
                    last_ai_msg = st.session_state.history.messages[-1]
                    tool_calls = last_ai_msg.tool_calls
                    streaming = False
                    tool_call = tool_calls[0]
                    if last_ai_msg.content is None or last_ai_msg.content == "":
                        additional_kwargs = tool_call_proc(
                            tool_call,
                            st.session_state.history.messages,
                            base64_image,
                            image_width,
                        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": content,
                "additional_kwargs": additional_kwargs,
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
    partial_run_agent = partial(run_agent, create_agent(used_model))
    agent_tools.extend([partial(load_urls_tool, used_model)])

    FUN_MAPPING["GenerateImageTool"] = partial_generate_image
    FUN_MAPPING["AnnotateImageTool"] = partial_annotate_image
    FUN_MAPPING["RunAgentTool"] = partial_run_agent
    used_model = used_model.bind_tools(
        [GenerateImageTool, AnnotateImageTool, RunAgentTool]
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

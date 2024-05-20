import asyncio
import base64
import os
import sys
from inspect import getframeinfo, stack
from typing import Any

import nest_asyncio
import streamlit as st

from functools import partial


from langchain import hub
from langchain.tools import tool
from langchain.agents import Tool, load_tools
from langchain.agents.agent import AgentExecutor
from langchain.agents.structured_chat.base import create_structured_chat_agent
from langchain.memory import ChatMessageHistory
from langchain.prompts import HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages.tool import ToolMessage, ToolMessageChunk

import requests
from PIL import Image, UnidentifiedImageError

from rich.pretty import pprint

VERBOSE = True
MAX_TOKEN = 2048

OPENAI_LLM = "gpt-4o"
GOOGLE_LLM = "gemini-1.5-flash-latest"

FUN_MAPPING = {}

nest_asyncio.apply()

st.set_page_config(layout="wide")


def pretty_print(title: str = "Untitled", content: Any = None):
    if not VERBOSE:
        return

    info = getframeinfo(stack()[1][0])
    print()
    pprint(
        f":--> {title} --> {info.filename} --> {info.function} --> line: {info.lineno} --:"
    )
    pprint(content)


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


def image_url_to_image(image_url: str) -> Image:
    return Image.open(requests.get(image_url, stream=True).raw)


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


class GenerateImageTool(BaseModel):
    """Generate an image to illustrate for user request."""

    context: str = Field(
        ...,
        description="The context for generating an image to illustrate what the user requested.",
    )


def chat_with_model(
    model: BaseChatModel,
    base64_image: bytes = None,
    streaming=False,
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
                st.image(image_url_to_image(additional_kwargs["image_url"]))

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
                content, additional_kwargs = None, None
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
                        tool_id, tool_name, context = (
                            tool_call["id"],
                            tool_call["name"],
                            tool_call["args"]["context"],
                        )
                        with st.spinner("Generating image..."):
                            try:
                                func = FUN_MAPPING.get(tool_name, None)
                                image, image_url = (
                                    func(context=context)
                                    if func
                                    else "No tool provided."
                                )
                                if image and image_url:
                                    st.image(image)
                                else:
                                    st.write("No image generated.")
                                # Finished tool call with a function, add result to history,
                                # The model needs it for future interaction.
                                additional_kwargs = (
                                    {"image_url": image_url} if image_url else {}
                                )
                                st.session_state.history.messages.append(
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
                                st.session_state.history.messages.append(
                                    ToolMessage(
                                        content=f"Tool was called but failed to generate image.\n\n{e}",
                                        tool_call_id=tool_id,
                                    )
                                )
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": content,
                "additional_kwargs": additional_kwargs,
            }
        )


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
    partial_generate_image = partial(
        generate_image,
        model=ChatOpenAI(
            model=OPENAI_LLM,
            temperature=st.session_state.key_temperature,
            max_tokens=MAX_TOKEN,
        ),
    )
    FUN_MAPPING["GenerateImageTool"] = partial_generate_image
    used_model = used_model.bind_tools([GenerateImageTool])
    chat_with_model(
        used_model,
        base64_image,
        streaming=st.session_state.get("key_streaming", True),
    )
    streaming = st.sidebar.checkbox("Streamming", True, key="key_streaming")


if __name__ == "__main__":
    asyncio.run(main())

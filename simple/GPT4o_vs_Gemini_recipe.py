import asyncio
import base64
import os
import sys
from functools import partial
from inspect import getframeinfo, stack
from typing import Any

import nest_asyncio
import streamlit as st
from langchain.memory import ChatMessageHistory
from langchain.prompts import HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from langchain_core.runnables import Runnable

from langchain import hub
from langchain.agents import Tool, load_tools
from langchain.agents.agent import AgentExecutor
from langchain.agents.structured_chat.base import create_structured_chat_agent

from langchain_core.pydantic_v1 import BaseModel, Field

from rich.pretty import pprint

nest_asyncio.apply()

st.set_page_config(layout="wide")

VERBOSE = True


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
    if st.session_state.model_sel == "Gemini Pro":
        # https://python.langchain.com/v0.1/docs/integrations/chat/google_generative_ai/#gemini-prompting-faqs
        # The Gemini hasn't supported multiturn approach which means a proper system and history places
        history = st.session_state.history
        prompt = ChatPromptTemplate.from_messages(
            [
                HumanMessagePromptTemplate.from_template(
                    template=(
                        [
                            {
                                "type": "text",
                                "text": f"Chat history:\n{history}"
                                + "\n\nMy question:\n\n{query}\n\n",
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
                                "text": f"Chat history:\n{history}"
                                + "\n\nMy question:\n\n{query}\n\n",
                                "text": "{query}",
                            },
                        ]
                    )
                ),
            ]
        )
    else:
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


def chat_with_model(model: BaseChatModel, base64_image: bytes = None, streaming=False):
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history" not in st.session_state:
        st.session_state.history = ChatMessageHistory()
    pretty_print("history", st.session_state.history)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if prompt := st.chat_input("Write...", key="chat_input"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chat_chain = RunnableWithMessageHistory(
                    create_chain(
                        model,  # .bind_tools([VisualRecipe]),
                        base64_image,
                    ),  # .with_listeners(on_end=fn_end),
                    lambda _: st.session_state.history,
                    input_messages_key="query",
                    history_messages_key="history",
                )
                out_met = chat_chain.invoke if not streaming else chat_chain.stream
                res = out_met(
                    {"query": prompt},
                    {"configurable": {"session_id": None}},
                )
                if not streaming:
                    content = res.content
                    st.write(content)
                else:
                    content = st.write_stream(res)

        st.session_state.messages.append({"role": "assistant", "content": content})


class VisualRecipe(BaseModel):
    """visualize the recipe"""

    context: str = Field(
        ...,
        description="""The prompt asks the model to generate an image based on an eating recipe to 
visualize the process of making the recipe.
""",
    )


@st.experimental_dialog("Generate image", width="large")
def gen_image(context: str, max_tokens: int):
    with st.spinner("Generating..."):
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=st.session_state.key_temperature,
            max_tokens=max_tokens,
        )
        tools = load_tools(["dalle-image-generator"])
        agent = create_structured_chat_agent(
            llm=llm,
            tools=tools,
            prompt=hub.pull("hwchase17/structured-chat-agent"),
        )
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )
        prompt = """Create an image with the following context:
    Context:
    {context}

    Notice: Return a markdown style image link.
    """
        image_gen = agent_executor.invoke({"input": prompt})
        st.markdown(f"![]({image_gen['output']})")
        pretty_print("Image Gen:", image_gen)


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
    max_tokens = 2048

    st.session_state["model_sel"] = st.sidebar.selectbox(
        "Model", ["GPT-4o", "Gemini Pro"], index=0
    )
    chat_with_model(
        (
            ChatGoogleGenerativeAI(
                model=(
                    "gemini-pro-vision" if base64_image else "gemini-pro"
                ),  # Gemini can currently only process text and text-image separately.
                temperature=st.session_state.key_temperature,
                max_tokens=max_tokens,
            )
            if st.session_state.model_sel == "Gemini Pro"
            else ChatOpenAI(
                model="gpt-4o",
                temperature=st.session_state.key_temperature,
                max_tokens=max_tokens,
            )
        ),
        base64_image,
        streaming=st.session_state.get("key_streaming", True),
    )
    streaming = st.sidebar.checkbox("Streamming", True, key="key_streaming")

    gen_img_prompt = st.sidebar.text_area("Generate image", placeholder="prompt...")
    if st.sidebar.button("Generate"):
        if gen_img_prompt is not None and gen_img_prompt != "":
            gen_image(gen_img_prompt, max_tokens)


if __name__ == "__main__":
    asyncio.run(main())

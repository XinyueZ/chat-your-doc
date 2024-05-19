import base64
import os
import sys
from functools import partial
from inspect import getframeinfo, stack
from typing import Any

import asyncio
import nest_asyncio

import streamlit as st
from langchain.chains import ConversationChain
from langchain.memory import ChatMessageHistory
from langchain.prompts import HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
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


def create_chain(model: BaseChatModel, base64_image: bytes):
    if st.session_state.model_sel == "Gemini-Pro-Vision":
        # https://python.langchain.com/v0.1/docs/integrations/chat/google_generative_ai/#gemini-prompting-faqs
        # The Gemini hasn't supported multiturn approach which means a proper system and history places
        prompt = ChatPromptTemplate.from_messages(
            [
                HumanMessagePromptTemplate.from_template(
                    template=(
                        [
                            {
                                "type": "text",
                                "text": f"Conversation history:\n\n{st.session_state.history}\n\n"
                                + "My question:\n\n{query}\n\n",
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
                if not streaming:
                    content = res.content
                    st.write(content)
                else:
                    content = st.write_stream(res)

        st.session_state.messages.append({"role": "assistant", "content": content})


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
        "Model", ["GPT-4o", "Gemini-Pro-Vision"], index=0
    )
    chat_with_model(
        (
            ChatGoogleGenerativeAI(
                model="gemini-pro-vision",
                temperature=st.session_state.key_temperature,
                max_tokens=max_tokens,
            )
            if st.session_state.model_sel == "Gemini-Pro-Vision"
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


if __name__ == "__main__":
    asyncio.run(main())

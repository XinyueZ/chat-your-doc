import streamlit as st
from typing import Any
from inspect import getframeinfo, stack
from rich.pretty import pprint
import os, sys
import base64
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.chains import ConversationChain

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


st.set_page_config(layout="wide")


def query_model(
    message: str,
    base64_image: bytes,
    streaming=True,
    temperature=0.0,
    max_tokens=2048 * 2,
):
    chat = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=1024 * 2)
    out_met = chat.invoke if not streaming else chat.stream
    res = out_met(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": message},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ]
            )
        ]
    )
    return res


def chat_with_model():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if prompt := st.chat_input("Write...", key="chat_input"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                res = st.session_state["bot"](prompt)
            try:
                content = query_model(prompt, st.session_state["base64_image"])
            except Exception as e:
                pretty_print("Cannot streaming", str(e))
                st.write(res.response)
                content = res.response
            pretty_print("content", content)
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


def main():
    base64_image = doc_uploader()
    if base64_image:
        st.sidebar.image(st.session_state["file_name"], use_column_width=True)
        if query := st.text_input(
            "Query",
            key="query_text",
            placeholder="Ask",
        ):
            response = query_model(query, base64_image)
            st.write_stream(response)


if __name__ == "__main__":
    main()

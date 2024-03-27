import os
from typing import Any, List, Literal

import streamlit as st
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.memory import ChatMessageHistory
from langchain.prompts import (
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough, chain
from langchain_core.runnables.base import RunnableSerializable
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

from loguru import logger
from rich.pretty import pprint

os.environ["LANGCHAIN_PROJECT"] = "nvidia_vs_groq"  # langsmith open


def pretty_print(title: str = None, content: Any = None):
    if title is None:
        print(content)
        return
    print(title)
    pprint(content)


st.session_state["history"] = (
    ChatMessageHistory()
    if "history" not in st.session_state
    else st.session_state["history"]
)

embedding = NVIDIAEmbeddings(model="nvolveqa_40k")

model_map = {
    "Groq Mixtral": "mixtral-8x7b-32768",
    "Groq LLaMA2": "llama2-70b-4096",
    "Nvidia Mixtral": "mixtral_8x7b",
    "Nvidia Llama2": "llama2_70b",
}

llm_temperature = st.sidebar.slider(
    "LLM temperature for RAG",
    0.0,
    1.0,
    0.5,
    key="llm_temperature_slider",
)


def llm_selector(model_name: str) -> BaseChatModel:
    model_brand = model_name.split(" ")[0]
    match model_brand:
        case "Groq":
            return ChatGroq(
                model=model_map[model_name],
                temperature=llm_temperature,
            )
        case "Nvidia":
            return ChatNVIDIA(
                model=model_map[model_name],
                temperature=llm_temperature,
            )
        case _:
            raise ValueError(f"Model {model_name} not found")


llm_name_selector = st.sidebar.selectbox(
    "Select LLM for RAG",
    model_map.keys(),
    index=2,
    key="llm_selector",
)
llm = llm_selector(llm_name_selector)


def create_retriever(file_path: str) -> BaseRetriever:
    loader = UnstructuredPDFLoader(file_path)
    docs = loader.load()
    chunks = SentenceTransformersTokenTextSplitter().split_documents(docs)
    db = FAISS.from_documents(chunks, embedding=embedding)
    return db.as_retriever()


def router_chain() -> RunnableSerializable:
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                """Given a conversation history to decide what next-step will follow, "standardalonequestion"  or "summary".
"standardalonequestion" if the conversation is just started.
"summary" if the conversation is ongoing (more than one interaction between the human and the AI).
Notice: Only return eithter "standardalonequestion" or "summary" without any instruction text, reasoning text, headlines, leading-text or other additional information.
"""
            ),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template(
                """What is the next step? (standardalonequestion or summary) and what is the conversation about based on history?"""
            ),
        ]
    )

    return prompt | llm | StrOutputParser()


def summary_chain() -> RunnableSerializable:
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                """Given a chat history and a follow-up question. Summerize the conversation based on the chat history and the follow-up question.
If the history is empty, do nothing just return the follow-up question as the summary.
Notice: Only return the final summary without any instruction text, headlines, leading-text or other additional information."""
            ),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
    )

    return prompt | llm | StrOutputParser()


def standalone_question_chain() -> RunnableSerializable:
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                """Given a chat history and a follow-up question, rephrase the follow-up question to be a standalone question. \
Do NOT answer the question, just reformulate it if needed, otherwise return it as is.
Notice: Only return the final standalone question without any instruction text, headlines, leading-text or other additional information."""
            ),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
    )

    return prompt | llm | StrOutputParser()


@chain
def route(info) -> RunnableSerializable:
    pretty_print("info", info)
    if "standardalonequestion" in info["next_step"]:
        pretty_print("standalone_question_chain", info["next_step"])
        return standalone_question_chain()

    if "summary" in info["next_step"]:
        pretty_print("summary", info["next_step"])
        return summary_chain()


def create_chain(
    base_retriever: BaseRetriever,
) -> RunnableSerializable:
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                "Answer question solely based on the following context:\n{context}\n"
            ),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
    )
    # RunnablePassthrough.assign(context=standalone_question_chain() | base_retriever)
    # RunnablePassthrough.assign(context=summary_chain() | base_retriever)
    mid_chain = (
        RunnablePassthrough.assign(next_step=router_chain())
        | RunnablePassthrough.assign(context=(route | base_retriever))
        | prompt
        | llm
        | StrOutputParser()
    )

    final_chain = RunnableWithMessageHistory(
        mid_chain,
        lambda _: st.session_state["history"],
        input_messages_key="question",
        history_messages_key="history",
    )
    return final_chain | StrOutputParser()


def doc_uploader() -> BaseRetriever | None:
    with st.sidebar:
        uploaded_doc = st.file_uploader(
            "# Upload one text content file", key="doc_uploader"
        )
        if not uploaded_doc:
            st.session_state["file_name"] = None
            st.session_state["queries"] = None
            logger.debug("No file uploaded")
            return None
        if uploaded_doc:
            tmp_dir = "tmp/"
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir)
            temp_file_path = os.path.join(tmp_dir, f"{uploaded_doc.name}")
            with open(temp_file_path, "wb") as file:
                file.write(uploaded_doc.getvalue())
                file_name = uploaded_doc.name
                logger.debug(f"Uploaded {file_name}")
                uploaded_doc.flush()
                uploaded_doc.close()

                # os.remove(temp_file_path)

                if st.session_state.get("file_name") == file_name:
                    logger.debug("Same file, same quiries, no indexing needed")
                    return st.session_state["retriever"]

                logger.debug("New file, new queries, indexing needed")
                st.session_state["file_name"] = file_name
                st.session_state["retriever"] = create_retriever(temp_file_path)
                return st.session_state["retriever"]
        return None


def main():
    st.sidebar.title("Upload file")
    st.sidebar.write(
        "[Suggest this file to ask some questions](https://dl.dropbox.com/scl/fi/xojn7rk5drda8ba4i90xr/4b1ca7c6-b279-4ed9-961a-484cadf8dd16.pdf?rlkey=aah3wklftddsgw7g5lrkv2tg4&dl=0)"
    )
    retriever: BaseRetriever | None = doc_uploader()
    if retriever is None:
        return
    question = st.text_input(
        "Question",
        key="question",
        placeholder="Enter your question here",
    ).strip()

    if question is not None and question != "":
        final_chain = create_chain(base_retriever=retriever)
        response = final_chain.stream(
            {"question": question},
            {"configurable": {"session_id": None}},
        )
        st.write_stream(response)
        st.sidebar.write("History")
        st.sidebar.write(st.session_state["history"].messages)


if __name__ == "__main__":
    main()

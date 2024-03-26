from multiprocessing import context
import os
from typing import Any, List, Tuple

import streamlit as st
from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.base import RunnableSerializable
from langchain_groq import ChatGroq
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from loguru import logger
from rich.pretty import pprint
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
)
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.prompts import (
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
)

os.environ["LANGCHAIN_PROJECT"] = "nvidia_vs_groq"  # langsmith open


def pretty_print(title: str = None, content: Any = None):
    if title is None:
        print(content)
        return
    print(title)
    pprint(content)


embedding = NVIDIAEmbeddings(model="nvolveqa_40k")

model_map = {
    "Groq Mixtral": "mixtral-8x7b-32768",
    "Groq LLaMA2": "llama2-70b-4096",
    "Nvidia Mixtral": "mixtral_8x7b",
    "Nvidia Llama2": "llama-2",
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
    key="llm_selector",
)
llm = llm_selector(llm_name_selector)


def create_retriever(file_path: str) -> BaseRetriever:
    loader = UnstructuredPDFLoader(file_path)
    docs = loader.load()
    chunks = SentenceTransformersTokenTextSplitter().split_documents(docs)
    db = FAISS.from_documents(chunks, embedding=embedding)
    return db.as_retriever()


def standalone_question_chain() -> RunnableSerializable:
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                """Given a chat history and a follow-up question, rephrase the follow-up question to be a standalone question. \
Do NOT answer the question, just reformulate it if needed, otherwise return it as is.
Only return the final standalone question."""
            ),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template(
                "<Question>{question}</Question>"
            ),
        ]
    )

    return prompt | llm


st.session_state["history"] = (
    ChatMessageHistory()
    if "history" not in st.session_state
    else st.session_state["history"]
)


def create_chain(
    base_retriever: BaseRetriever,
) -> RunnableSerializable:
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                "Answer question solely based on the following context:\n<Documents>\n{context}\n</Documents>"
            ),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("</Question>{question}</Question>"),
        ]
    )

    chain = (
        RunnablePassthrough.assign(
            context=standalone_question_chain() | StrOutputParser() | base_retriever
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    final_chain = RunnableWithMessageHistory(
        chain,
        lambda _: st.session_state["history"],
        input_messages_key="question",
        history_messages_key="history",
    )
    return final_chain


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
    )

    if question is not None and question != "":
        chain = create_chain(base_retriever=retriever)
        response = (chain | StrOutputParser()).stream(
            {"question": question},
            {"configurable": {"session_id": None}},
        )
        st.write_stream(response)
        st.sidebar.write("History")
        st.sidebar.write(st.session_state["history"].messages)


if __name__ == "__main__":
    main()

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
    0.0,
    key="llm_temperature_slider",
)

hypo_gen_temperature = st.sidebar.slider(
    "HyDE temperature",
    0.0,
    1.0,
    0.8,
    key="hypo_gen_temperature_slider",
)


def llm_selector(
    model_name: str,
    temperature: float,
) -> BaseChatModel:
    model_brand = model_name.split(" ")[0]
    match model_brand:
        case "Groq":
            return ChatGroq(
                model=model_map[model_name],
                temperature=temperature,
                max_tokens=2048 * 2,
            )
        case "Nvidia":
            return ChatNVIDIA(
                model=model_map[model_name],
                temperature=temperature,
                max_tokens=2048 * 2,
            )
        case _:
            raise ValueError(f"Model {model_name} not found")


llm_name_selector = st.sidebar.selectbox(
    "Select LLM for RAG",
    model_map.keys(),
    index=2,
    key="llm_selector",
)
llm = llm_selector(llm_name_selector, llm_temperature)
hyde_llm = llm_selector(llm_name_selector, hypo_gen_temperature)


def create_retriever(file_path: str) -> BaseRetriever:
    loader = UnstructuredPDFLoader(file_path)
    docs = loader.load()
    chunks = SentenceTransformersTokenTextSplitter().split_documents(docs)
    db = FAISS.from_documents(chunks, embedding=embedding)
    return db.as_retriever()


def route_chain() -> RunnableSerializable:
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                """As an AI assistant for application route, determine whether to enter the process for handling declarative and normal sentences or interrogative and question sentences (indirect or direct questions) based on the user's query tone inside [Query] marks,
If it is a declarative and normal sentence process, return the string: "hydechain"; 
otherwise, return: "standardalonequery".

Notice: 
Do not answer the query or make up the query, only return as simple as possible, eithter "standardalonequery" or "hydechain" as string without any instruction text, reasoning text, headlines, leading-text or other additional information.
"""
            ),
            HumanMessagePromptTemplate.from_template("""[Query]{query}[Query]"""),
        ]
    )

    return prompt | llm | StrOutputParser()


def hyde_chain() -> RunnableSerializable:
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                """Generate a different version of the text inside [Origin text] marks and conversation history to retrieve relevant documents from a vector database.
The version is for better model comprehension while maintaining the original text sentiment and brevity.
Your goal is to help the user overcome some of the limitations of the distance-based similarity search. 
Notice: Only return the reformulated statement without any explaination or additional information."""
            ),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template(
                "[Origin text]\n{query}\n[Origin text]"
            ),
        ]
    )

    return prompt | hyde_llm | StrOutputParser()


def standalone_query_chain() -> RunnableSerializable:
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                """Given a conversation history and a follow-up query inside [Query] marks, rephrase the follow-up query to be a standalone query. \
Do NOT answer the query, just reformulate it if needed, otherwise return it as is.
Notice: Only return the final standalone query without any instruction text, headlines, leading-text or other additional information."""
            ),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("[Query]{query}[Query]"),
        ]
    )

    return prompt | llm | StrOutputParser()


def build_final_chain(
    base_retriever: BaseRetriever,
) -> RunnableSerializable:
    @chain
    def _routed_chain_(info) -> RunnableSerializable:
        pretty_print("info", info)
        if "standardalonequery" in info["next_step"].lower():
            pretty_print("standalone_query_chain")
            return standalone_query_chain()

        if "hydechain" in info["next_step"].lower():
            pretty_print("hydechain")
            return hyde_chain()

        raise ValueError("Invalid next step")

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                "Answer query inside [Query] marks solely based on the following context:\n{context}\n"
            ),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("[Query]{query}[Query]"),
        ]
    )

    mid_chain = (
        RunnablePassthrough.assign(next_step=route_chain())
        | RunnablePassthrough.assign(context=(_routed_chain_ | base_retriever))
        | prompt
        | llm
        | StrOutputParser()
    )

    final_chain = RunnableWithMessageHistory(
        mid_chain,
        lambda _: st.session_state["history"],
        input_messages_key="query",
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
        "[Suggest this file to query](https://dl.dropbox.com/scl/fi/xojn7rk5drda8ba4i90xr/4b1ca7c6-b279-4ed9-961a-484cadf8dd16.pdf?rlkey=aah3wklftddsgw7g5lrkv2tg4&dl=0)"
    )
    retriever: BaseRetriever | None = doc_uploader()
    if retriever is None:
        return
    query = st.text_input(
        "Query",
        key="query",
        placeholder="Enter your query here",
    ).strip()

    if query is not None and query != "":
        final_chain = build_final_chain(base_retriever=retriever)
        response = final_chain.stream(
            {"query": query},
            {"configurable": {"session_id": None}},
        )
        st.write_stream(response)
        st.sidebar.write("History")
        st.sidebar.write(st.session_state["history"].messages)


if __name__ == "__main__":
    main()

from distutils.command import upload
from math import e
import os
from pathlib import Path
from typing import List, Tuple
from numpy import place

import streamlit as st
from loguru import logger
from pydantic import FilePath

from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import UnstructuredPDFLoader, PyPDFLoader
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain_core.documents import Document

K = 5


class BaseQuerier:
    def query(self, query_text: str) -> str:
        return f"""{query_text}

        Only answer based on the context you have, don't use any external information to makeup the answer."""


class LangChainQuerier(BaseQuerier):
    def __init__(self, file_path: FilePath) -> None:
        def load_and_split(path: str) -> List[Document]:
            loader = UnstructuredPDFLoader(path)
            docs = loader.load()
            text_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=100)
            texts = text_splitter.split_documents(docs)
            return texts

        chunks: List[Document] = load_and_split(path=str(file_path))
        self.vector_store = Chroma.from_documents(chunks, embedding=OpenAIEmbeddings())

    def query(self, query_text: str) -> str:
        updated_query_text = super().query(query_text)
        relevant_docs: List[Document] = self.vector_store.similarity_search(
            query_text, K
        )
        qa_chain = load_qa_chain(
            ChatOpenAI(temperature=0, model="gpt-4-0125-preview"),
            chain_type="refine",
            verbose=True,
        )
        result = qa_chain.run(
            input_documents=relevant_docs, question=updated_query_text
        )
        return result


class LlamaIndexQuerier(BaseQuerier):
    def __init__(self, file_path: FilePath) -> None:
        pass

    def query(self, query_text: str) -> str:
        pass


def doc_uploader() -> Tuple[BaseQuerier] | None:
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
                    return st.session_state["queries"]

                logger.debug("New file, new queries, indexing needed")
                st.session_state["file_name"] = file_name
                st.session_state["queries"] = (
                    LangChainQuerier(Path(temp_file_path)),
                    LlamaIndexQuerier(Path(temp_file_path)),
                )
                return st.session_state["queries"]
        return None


def main():
    def clear_query_input():
        st.session_state["query_input"] = ""

    st.sidebar.radio(
        "Method",
        ["Refine(LangChain)", "MultiStepQueryEngine(Llama-Index)"],
        index=1,
        key="method_selector",
        on_change=clear_query_input,
    )

    queries: Tuple[BaseQuerier] = doc_uploader()

    if queries == None:
        return

    lc_querier, lli_querier = queries

    query_text = st.text_input(
        "Query", key="query_input", placeholder="Enter your query here"
    )

    if query_text is not None and query_text != "":
        if st.session_state.method_selector == "Refine(LangChain)":
            result: str = lc_querier.query(query_text)
        else:
            result: str = "hello result"
        st.title("Result")
        st.write(result)


if __name__ == "__main__":
    main()

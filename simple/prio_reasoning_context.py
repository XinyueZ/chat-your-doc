import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import streamlit as st
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.embeddings.utils import EmbedType
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.indices.base import BaseIndex
from llama_index.core.indices.query.query_transform.base import (
    StepDecomposeQueryTransform,
)
from llama_index.core.llms.utils import LLMType
from llama_index.core.node_parser import SentenceSplitter, SentenceWindowNodeParser
from llama_index.core.query_engine import MultiStepQueryEngine
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.service_context import ServiceContext
from llama_index.legacy.core.response.schema import RESPONSE_TYPE
from llama_index.llms.openai import OpenAI
from loguru import logger
from pydantic import FilePath

K = 5


class BaseQuerier:
    def __init__(self, **kwargs) -> None:
        logger.debug(f"Querier initialized with {kwargs}")
        self.temperature = kwargs.get("temperature", 1.5)

    def get_intermediate_information(self) -> Tuple[str]:
        raise NotImplementedError

    def query(self, query_text: str) -> str:
        return f"""{query_text}

        Only answer based on the context you have, don't use any external or additional information to makeup the answer."""


class LangChainQuerier(BaseQuerier):
    def __init__(self, file_path: FilePath, **kwargs) -> None:
        super().__init__(**kwargs)

        def load_and_split(path: str) -> List[Document]:
            loader = UnstructuredPDFLoader(path)
            docs = loader.load()
            text_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=100)
            texts = text_splitter.split_documents(docs)
            return texts

        chunks: List[Document] = load_and_split(path=str(file_path))
        self.vector_store = Chroma.from_documents(chunks, embedding=OpenAIEmbeddings())
        self.model = ChatOpenAI(
            temperature=self.temperature,
            model="gpt-4-0125-preview",
        )

    def query(self, query_text: str) -> str:
        updated_query_text = super().query(query_text)
        relevant_docs: List[Document] = self.vector_store.similarity_search(
            query_text, K
        )
        qa_chain = load_qa_chain(
            self.model,
            chain_type="refine",
            verbose=True,
        )
        result = qa_chain.run(
            input_documents=relevant_docs, question=updated_query_text
        )
        return result

    def get_intermediate_information(self) -> Tuple[str]:
        return ()


class LlamaIndexQuerier(BaseQuerier):
    def __init__(self, file_path: FilePath, **kwargs) -> None:
        super().__init__(**kwargs)
        self.docs: SimpleDirectoryReader = SimpleDirectoryReader(
            input_files=[str(file_path)]
        ).load_data()
        self.model: OpenAI = OpenAI(
            temperature=self.temperature,
            model="gpt-4-0125-preview",
        )
        embs = "local:BAAI/bge-small-en-v1.5"
        service_context: ServiceContext = self.create_service_context(self.model, embs)
        vector_index: BaseIndex = VectorStoreIndex.from_documents(
            self.docs,
            service_context=service_context,
            show_progress=True,
            transformations=[SentenceSplitter()],
        )
        step_decompose_transform = StepDecomposeQueryTransform(
            llm=self.model, verbose=True
        )
        base_query_engine: BaseQueryEngine = vector_index.as_query_engine()
        self.query_engine = MultiStepQueryEngine(
            query_engine=base_query_engine,
            query_transform=step_decompose_transform,
            index_summary="Used to answer questions about what the user queries.",
        )

    def create_service_context(self, llm: LLMType, embs: EmbedType) -> ServiceContext:
        return ServiceContext.from_defaults(
            llm=self.model,
            embed_model=embs,
        )

    def query(self, query_text: str) -> str:
        self.res = self.query_engine.query(super().query(query_text))
        return self.res.response

    def get_intermediate_information(self) -> Tuple[str]:
        sub_qa: Dict[str, Any] = self.res.metadata["sub_qa"]
        sub_qa_list: List[str] = tuple(
            [
                "**Question:**\n{}\n\n**Answer:**\n{}\n\n".format(t[0], t[1].response)
                for t in sub_qa
            ]
        )
        return sub_qa_list


def doc_uploader(temperature: float) -> Tuple[BaseQuerier] | None:
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
                    LangChainQuerier(Path(temp_file_path), temperature=temperature),
                    LlamaIndexQuerier(Path(temp_file_path), temperature=temperature),
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

    st.sidebar.write("##### Try to play with this doc:")
    st.sidebar.write(
        "[Paper about Vector Search with OpenAI Embeddings: Lucene Is All You Need](https://dl.dropbox.com/scl/fi/xojn7rk5drda8ba4i90xr/4b1ca7c6-b279-4ed9-961a-484cadf8dd16.pdf?rlkey=aah3wklftddsgw7g5lrkv2tg4&dl=0)"
    )

    temperature: float = st.sidebar.slider(
        "Tempetrature",
        0.0,
        1.8,
        1.5,
        key="temperature_slider",
    )

    queries: Tuple[BaseQuerier] = doc_uploader(temperature=temperature)

    if queries is None:
        return

    lc_querier, lli_querier = queries[0], queries[1]

    query_text = st.text_input(
        "Query",
        key="query_text",
        placeholder="Enter your query here",
        value="What is the Vector Search?",
    )

    if query_text is not None and query_text != "":
        if st.session_state.method_selector == "Refine(LangChain)":
            querier = lc_querier
        else:
            querier = lli_querier
        result: str = querier.query(query_text)
        inter_info = querier.get_intermediate_information()
        with st.expander("Sub Q&A"):
            for info in inter_info:
                st.write(info)
        st.title("Result")
        st.write(result)


if __name__ == "__main__":
    main()

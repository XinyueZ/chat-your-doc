import os
from pathlib import Path
from typing import List

import chromadb
import streamlit as st
from chromadb.api.models.Collection import Collection
from llama_index import (
    ServiceContext,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    get_response_synthesizer,
)
from llama_index.response_synthesizers.base import BaseSynthesizer
from llama_index.response_synthesizers.type import ResponseMode
from llama_index.postprocessor.types import BaseNodePostprocessor
from llama_index.indices.postprocessor import (
    MetadataReplacementPostProcessor,
    SentenceTransformerRerank,
)
from llama_index.core import BaseQueryEngine, BaseRetriever
from llama_index.agent import OpenAIAgent
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.embeddings.utils import EmbedType
from llama_index.indices.base import BaseIndex
from llama_index.indices.document_summary import DocumentSummaryIndex
from llama_index.llms import OpenAI
from llama_index.llms.utils import LLMType
from llama_index.node_parser import SentenceWindowNodeParser
from llama_index.response.schema import RESPONSE_TYPE
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.vector_stores import ChromaVectorStore
from loguru import logger
from pydantic import FilePath
from llama_index.indices.document_summary import (
    DocumentSummaryIndex,
    DocumentSummaryIndexLLMRetriever,
)

TEMPERATURE = 0.0
SIM_TOP_K = 3
RERANK_TOP_K = 3
CHUNK_OVERLAP = 30
CHUNK_SIZE = 150
WIN_SZ = 3


def create_vectors(path: str, collection_name="tmp_collection") -> Collection:
    chroma_client = chromadb.PersistentClient(path)
    # https://github.com/run-llama/llama_index/issues/6528
    return chroma_client.get_or_create_collection(collection_name)


class LlamaIndexMultiVectorSummaryAgent:
    def __init__(self) -> None:
        if "agent" not in st.session_state:
            filepath = self._upload_doc()
            if filepath and os.path.exists(filepath):
                logger.debug(f"Loaded Filepath: {filepath}")

                llm = OpenAI(model="gpt-4-1106-preview", temperature=TEMPERATURE)
                embs = "local:BAAI/bge-small-en-v1.5"

                service_context: ServiceContext = (
                    LlamaIndexMultiVectorSummaryAgent.create_service_context(llm, embs)
                )

                storage_context: StorageContext = (
                    LlamaIndexMultiVectorSummaryAgent.create_storage_context()
                )

                input_files: List[str] = [filepath]
                docs: SimpleDirectoryReader = SimpleDirectoryReader(
                    input_files=input_files,
                ).load_data()

                logger.debug("Start loading document index from storage")
                summary_index: BaseIndex = DocumentSummaryIndex.from_documents(
                    docs,
                    storage_context=storage_context,
                    service_context=service_context,
                    show_progress=True,
                )

                vector_index: BaseIndex = VectorStoreIndex.from_documents(
                    docs,
                    service_context=service_context,
                    storage_context=storage_context,
                    show_progress=True,
                )
                logger.debug("Finish loading document index from storage")

                logger.debug("Start creating agent with tools")
                query_engine_tools = [
                    QueryEngineTool(
                        query_engine=LlamaIndexMultiVectorSummaryAgent._from_retriever_to_query_engine(
                            service_context=service_context,
                            retriever=DocumentSummaryIndexLLMRetriever(
                                summary_index,
                                similarity_top_k=SIM_TOP_K,
                            ),
                        ),
                        metadata=ToolMetadata(
                            name=f"vector_tool",
                            description=f"Useful for questions related to specific facts in the `{filepath}` document",
                        ),
                    ),
                    QueryEngineTool(
                        query_engine=LlamaIndexMultiVectorSummaryAgent._from_retriever_to_query_engine(
                            service_context=service_context,
                            retriever=vector_index.as_retriever(
                                similarity_top_k=SIM_TOP_K
                            ),
                        ),
                        metadata=ToolMetadata(
                            name=f"summary_tool",
                            description=f"Useful for summarization questions about the `{filepath}` document",
                        ),
                    ),
                ]
                st.session_state["agent"] = OpenAIAgent.from_tools(
                    query_engine_tools,
                    llm=llm,
                    verbose=True,
                    system_prompt=f"""\
            You are a specialized agent designed to answer queries about the `{filepath}` document.
            You must ALWAYS use at least one of the tools provided when answering a question; do NOT rely on prior knowledge.\
            """,
                )
                logger.debug("Finish creating agent with tools")
                st.experimental_rerun()
        else:
            if st.sidebar.button("Reload file"):
                st.session_state.clear()
                st.experimental_rerun()

    def _upload_doc(self) -> FilePath | None:
        with st.sidebar:
            uploaded_doc = st.file_uploader("# Upload one PDF", key="doc_uploader")
            if not uploaded_doc:
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
                    return Path(temp_file_path)
            return None

    @classmethod
    def create_service_context(cls, llm: LLMType, embs: EmbedType) -> ServiceContext:
        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=WIN_SZ,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )

        return ServiceContext.from_defaults(
            # chunk_overlap=CHUNK_OVERLAP,
            # chunk_size=CHUNK_SIZE,
            node_parser=node_parser,
            llm=llm,
            embed_model=embs,
        )

    @classmethod
    def create_storage_context(cls) -> StorageContext:
        path: str = "./db/LlamaIndexMultiVectorSummary"
        return StorageContext.from_defaults(
            vector_store=ChromaVectorStore(create_vectors(path=path))
        )

    @classmethod
    def _from_retriever_to_query_engine(
        cls,
        service_context: ServiceContext,
        retriever: BaseRetriever,
    ) -> BaseQueryEngine:
        postproc: BaseNodePostprocessor = MetadataReplacementPostProcessor(
            target_metadata_key="window"
        )
        rerank: BaseNodePostprocessor = SentenceTransformerRerank(
            top_n=RERANK_TOP_K, model="BAAI/bge-reranker-base"
        )
        response_synthesizer: BaseSynthesizer = get_response_synthesizer(
            service_context=service_context,
            response_mode=ResponseMode.REFINE,
        )
        return RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[postproc, rerank],
        )

    def run(self):
        if "agent" in st.session_state:
            query: str = st.text_input("Query", placeholder="Enter query here")
            if query != "":
                agent: OpenAIAgent = st.session_state["agent"]
                result: RESPONSE_TYPE = agent.query(query)
                st.write(result.response)

    def __call__(self):
        self.run()


if __name__ == "__main__":
    LlamaIndexMultiVectorSummaryAgent()()

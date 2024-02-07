import os
from typing import List

import chromadb
import streamlit as st
from chromadb.api.models.Collection import Collection
from llama_index import (
    QueryBundle,
    ServiceContext,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    get_response_synthesizer,
)
from llama_index.core import BaseQueryEngine, BaseRetriever
from llama_index.embeddings import OpenAIEmbedding
from llama_index.embeddings.openai import OpenAIEmbeddingModelType
from llama_index.embeddings.utils import EmbedType
from llama_index.indices.base import BaseIndex
from llama_index.indices.document_summary import (
    DocumentSummaryIndex,
    DocumentSummaryIndexEmbeddingRetriever,
    DocumentSummaryIndexLLMRetriever,
)
from llama_index.indices.postprocessor import (
    MetadataReplacementPostProcessor,
    SentenceTransformerRerank,
)
from llama_index.llms import OpenAI
from llama_index.llms.utils import LLMType
from llama_index.node_parser import SentenceWindowNodeParser
from llama_index.postprocessor.types import BaseNodePostprocessor
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response.schema import RESPONSE_TYPE
from llama_index.response_synthesizers.base import BaseSynthesizer
from llama_index.response_synthesizers.type import ResponseMode
from llama_index.schema import NodeWithScore
from llama_index.vector_stores import ChromaVectorStore
from loguru import logger

MODE = "OR"
TEMPERATURE = 0.0
SIM_TOP_K = 3
RERANK_TOP_K = 3
CHUNK_OVERLAP = 30
CHUNK_SIZE = 150
WIN_SZ = 3
DOC_DIR = "./tmp"


def create_vectors(path: str, collection_name="tmp_collection") -> Collection:
    chroma_client = chromadb.PersistentClient(path)
    # https://github.com/run-llama/llama_index/issues/6528
    return chroma_client.get_or_create_collection(collection_name)


class MultiVectorSummaryRetriever(BaseRetriever):
    def __init__(
        self,
        summary_retriever: BaseRetriever,
        vector_retriever: BaseRetriever,
        mode: str = "OR",
    ) -> None:
        """Init params."""

        self._summary_retriever = summary_retriever
        self._vector_retriever = vector_retriever
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        summary_nodes = self._summary_retriever.retrieve(query_bundle)
        vector_nodes = self._vector_retriever.retrieve(query_bundle)

        summary_ids = {n.node.node_id for n in summary_nodes}
        vector_ids = {n.node.node_id for n in vector_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in summary_nodes})

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(summary_ids)
        else:
            retrieve_ids = vector_ids.union(summary_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes


class LlamaIndexVectorSummaryRetriever:
    _query_engine: BaseQueryEngine

    def __init__(self):
        if "query_engine" not in st.session_state:
            llm = OpenAI(model="gpt-4-1106-preview", temperature=TEMPERATURE)
            embs = "local:BAAI/bge-small-en-v1.5"  # OpenAIEmbedding(model=OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002)

            service_context: ServiceContext = (
                LlamaIndexVectorSummaryRetriever.create_service_context(llm, embs)
            )

            storage_context: StorageContext = (
                LlamaIndexVectorSummaryRetriever.create_storage_context()
            )

            required_exts: List[str] = [".pdf"]
            # Create a directory and put some files in it for querying.
            input_files: List[str] = [os.path.join(DOC_DIR, "track-anything.pdf")]
            docs: SimpleDirectoryReader = SimpleDirectoryReader(
                DOC_DIR,
                required_exts=required_exts,
                input_files=input_files,
            ).load_data()

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

            logger.info("Loading index from storage")

            # summary_index.as_retriever() will be replaced by
            summary_retriever: BaseRetriever = DocumentSummaryIndexLLMRetriever(
                summary_index,
                similarity_top_k=SIM_TOP_K,
            )
            vector_retriever: BaseRetriever = vector_index.as_retriever(
                similarity_top_k=SIM_TOP_K,
            )

            multi_vec_sum_retriever: BaseRetriever = MultiVectorSummaryRetriever(
                summary_retriever=summary_retriever,
                vector_retriever=vector_retriever,
                mode=MODE,
            )
            response_synthesizer: BaseSynthesizer = get_response_synthesizer(
                service_context=service_context,
                response_mode=ResponseMode.REFINE,
            )

            postproc: BaseNodePostprocessor = MetadataReplacementPostProcessor(
                target_metadata_key="window"
            )
            rerank: BaseNodePostprocessor = SentenceTransformerRerank(
                top_n=RERANK_TOP_K, model="BAAI/bge-reranker-base"
            )

            st.session_state["query_engine"] = RetrieverQueryEngine(
                retriever=multi_vec_sum_retriever,
                response_synthesizer=response_synthesizer,
                node_postprocessors=[postproc, rerank],
            )

        self._query_engine = st.session_state["query_engine"]

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

    def run(self):
        query: str = st.text_input("Query", placeholder="Enter query here")
        if query != "":
            result: RESPONSE_TYPE = self._query_engine.query(query)
            st.write(result.response)

    def __call__(self):
        self.run()


if __name__ == "__main__":
    LlamaIndexVectorSummaryRetriever()()

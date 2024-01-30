import chromadb
from llama_index.llms import OpenAI
import streamlit as st
from chromadb.api.models.Collection import Collection
from llama_index import (
    ServiceContext,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core import BaseQueryEngine
from llama_index.response.schema import RESPONSE_TYPE
from llama_index.vector_stores import ChromaVectorStore


def create_vectors(collection_name="tmp_collection") -> Collection:
    chroma_client = chromadb.PersistentClient("./db")
    return chroma_client.get_or_create_collection(collection_name)


class LlamaIndexContext:
    _query_engine: BaseQueryEngine

    def __init__(self):
        if "query_engine" not in st.session_state:
            service_context = LlamaIndexContext.create_service_context()
            storage_context = LlamaIndexContext.create_storage_context()
            index: VectorStoreIndex = VectorStoreIndex.from_documents(
                documents=SimpleDirectoryReader("./tmp").load_data(),
                service_context=service_context,
                storage_context=storage_context,
                show_progress=True,
            )
            st.session_state["query_engine"] = index.as_query_engine()
        self._query_engine = st.session_state["query_engine"]

    @classmethod
    def create_service_context(cls) -> ServiceContext:
        return ServiceContext.from_defaults(
            chunk_overlap=0,
            chunk_size=500,
            llm=OpenAI(),
        )

    @classmethod
    def create_storage_context(cls) -> StorageContext:
        return StorageContext.from_defaults(
            vector_store=ChromaVectorStore(create_vectors())
        )

    def run(self):
        query: str = st.text_input("Query", placeholder="Enter query here")
        if query != "":
            result: RESPONSE_TYPE = self._query_engine.query(query)
            st.write(result.response)

    def __call__(self):
        self.run()


if __name__ == "__main__":
    LlamaIndexContext()()

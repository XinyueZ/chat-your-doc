import streamlit as st
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import RESPONSE_TYPE


class HelloLlamaIndex:
    _query_engine: BaseQueryEngine

    def __init__(self):
        if "query_engine" not in st.session_state:
            index: VectorStoreIndex = VectorStoreIndex.from_documents(
                SimpleDirectoryReader("./tmp").load_data()
            )
            st.session_state["query_engine"] = index.as_query_engine()
        self._query_engine = st.session_state["query_engine"]

    def run(self):
        query: str = st.text_input("Query", placeholder="Enter your query here")
        if query != "":
            result: RESPONSE_TYPE = self._query_engine.query(query)
            st.write(result.response)

    def __call__(self):
        self.run()


if __name__ == "__main__":
    HelloLlamaIndex()()

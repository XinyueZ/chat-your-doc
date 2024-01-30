import streamlit as st
from llama_index import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import BaseQueryEngine
from llama_index.response.schema import RESPONSE_TYPE


class HelloLlamaIndex:
    _query_engine: BaseQueryEngine

    def __init__(self):
        index: VectorStoreIndex = VectorStoreIndex.from_documents(
            SimpleDirectoryReader("./tmp").load_data()
        )
        self._query_engine = index.as_query_engine()

    def run(self):
        query: str = st.text_input("Query", "hello world")
        if query != "":
            result: RESPONSE_TYPE = self._query_engine.query(query)
            st.write(result.response)

    def __call__(self):
        self.run()


if __name__ == "__main__":
    HelloLlamaIndex()()

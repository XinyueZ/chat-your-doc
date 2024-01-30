from typing import List, Type
from numpy import place

import streamlit as st
from llama_index import VectorStoreIndex, download_loader
from llama_index.core import BaseQueryEngine
from llama_index.readers.base import BaseReader
from llama_index.response.schema import RESPONSE_TYPE
from llama_index.schema import Document


class LlamaIndexHubSimple:
    _query_engine: BaseQueryEngine

    def __init__(self):
        if "query_engine" not in st.session_state:
            MarkdownReader: Type[BaseReader] = download_loader("MarkdownReader")
            reader = MarkdownReader()
            docs: List[Document] = reader.load_data("./README.md")

            index: VectorStoreIndex = VectorStoreIndex.from_documents(docs)
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
    LlamaIndexHubSimple()()

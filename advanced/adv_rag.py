import os
import uuid
from pathlib import Path
from typing import Any

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.chat import ChatPromptTemplate
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema.document import Document
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.storage import InMemoryStore
from langchain.vectorstores import Chroma
from langchain_core.retrievers import BaseRetriever
from loguru import logger
from pydantic import BaseModel, FilePath
from unstructured.documents.elements import Element
from unstructured.partition.pdf import partition_pdf


class DocElement(BaseModel):
    type: str
    content: Any


class DocElements(BaseModel):
    text_doc_elements: list[DocElement]
    table_doc_elements: list[DocElement]


class DocSummaries(BaseModel):
    text_doc_summaries: list[str]
    table_doc_summaries: list[str]


class SummaryCreator:
    def __call__(self, elements: list[Element]) -> DocElements:
        return self.build(elements)

    def create(self, elements: list[Element]) -> DocElements:
        text_doc_elements, table_doc_elements = list(), list()

        for element in elements:
            if "unstructured.documents.elements.Table" in str(type(element)):
                # logger.debug(f"Table: {element}")
                table_doc_elements.append(
                    DocElement(type="table", content=str(element))
                )
            elif "unstructured.documents.elements.CompositeElement" in str(
                type(element)
            ):
                # logger.debug(f"Composite element: {element}")
                text_doc_elements.append(DocElement(type="text", content=str(element)))
        return DocElements(
            text_doc_elements=text_doc_elements, table_doc_elements=table_doc_elements
        )


class SummaryChain:
    _prompt_text = """
    You should concisely summary content chunk:

    content: {content}
    """

    def __init__(self):
        prompt = ChatPromptTemplate.from_template(self._prompt_text)
        self._summarize_chain = (
            {"content": RunnablePassthrough()}
            | prompt
            | ChatOpenAI(temperature=0, model="gpt-4-1106-preview")
            | StrOutputParser()
        )

    def __call__(
        self, doc_elements: DocElements, max_con_curr: int = 5
    ) -> DocSummaries:
        return self.execute(doc_elements, max_con_curr)

    def execute(
        self, doc_elements: DocElements, max_con_curr: int = 10
    ) -> DocSummaries:
        texts = list(map(lambda e: e.content, doc_elements.text_doc_elements))
        text_doc_summaries = self._summarize_chain.batch(
            texts, {"max_concurrency": max_con_curr}
        )
        tables = list(map(lambda e: e.content, doc_elements.table_doc_elements))
        table_doc_summaries = self._summarize_chain.batch(
            tables, {"max_concurrency": max_con_curr}
        )
        return DocSummaries(
            text_doc_summaries=text_doc_summaries,
            table_doc_summaries=table_doc_summaries,
        )


class RetrieverBuilder:
    _key = "doc_key"
    _doc_store = InMemoryStore()

    def __call__(
        self, doc_elements: DocElements, doc_summaries: DocSummaries
    ) -> BaseRetriever:
        return self.build(doc_elements, doc_summaries)

    def build(
        self, doc_elements: DocElements, doc_summaries: DocSummaries
    ) -> BaseRetriever:
        assert len(doc_elements.text_doc_elements) == len(
            doc_summaries.text_doc_summaries
        ), "Text elements and summaries must be same length"
        assert len(doc_elements.table_doc_elements) == len(
            doc_summaries.table_doc_summaries
        ), "Table elements and summaries must be same length"

        text_keys = [str(uuid.uuid4()) for _ in doc_elements.text_doc_elements]
        table_keys = [str(uuid.uuid4()) for _ in doc_elements.table_doc_elements]

        retriever = MultiVectorRetriever(
            vectorstore=Chroma(
                collection_name="summaries",
                embedding_function=OpenAIEmbeddings(),
            ),
            docstore=self._doc_store,
            id_key=self._key,
        )

        retriever.vectorstore.add_documents(
            list(
                map(
                    lambda x: Document(metadata={self._key: x[0]}, page_content=x[1]),
                    list(zip(text_keys, doc_summaries.text_doc_summaries)),
                )
            )
        )
        retriever.docstore.mset(
            list(
                zip(
                    text_keys,
                    list(map(lambda x: x.content, doc_elements.text_doc_elements)),
                )
            )
        )
        retriever.vectorstore.add_documents(
            list(
                map(
                    lambda x: Document(metadata={self._key: x[0]}, page_content=x[1]),
                    list(zip(table_keys, doc_summaries.table_doc_summaries)),
                )
            )
        )
        retriever.docstore.mset(
            list(
                zip(
                    table_keys,
                    list(map(lambda x: x.content, doc_elements.table_doc_elements)),
                )
            )
        )
        return retriever


class QueryChain:
    _prompt_text = """
    Answer the question based on the context:

    context: {context}

    question: {question}
    """

    def __init__(self, retriever: BaseRetriever):
        prompt = ChatPromptTemplate.from_template(self._prompt_text)
        self._summarize_chain = (
            {"question": RunnablePassthrough(), "context": retriever}
            | prompt
            | ChatOpenAI(temperature=0, model="gpt-4-1106-preview")
            | StrOutputParser()
        )

    def __call__(self, query: str) -> str:
        return self.invoke(query)

    def invoke(self, query: str) -> str:
        return self._summarize_chain.invoke(query)


class App:
    def _read_pdf(
        self,
        filepath: FilePath | None,
        extract_images_in_pdf: bool,
        infer_table_structure: bool,
    ) -> list[Element] | None:
        if not filepath:
            return None
        raw_pdf_elements = partition_pdf(
            filename=filepath.absolute(),
            extract_images_in_pdf=extract_images_in_pdf,
            infer_table_structure=infer_table_structure,
            chunking_strategy="by_title",
            max_characters=4000,
            new_after_n_chars=3800,
            combine_text_under_n_chars=2000,
            image_output_dir_path="./tmp",
        )
        return raw_pdf_elements

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

    def __call__(self):
        st.title("# Advanced RAG")
        st.write(
            f"Data source: [Baidu Inc. 2023 Q3 Financial Reports](https://ir.baidu.com/static-files/f4006310-1a98-4d86-89a1-edfea1ef5d0e)"
        )
        st.write(
            f"Data source: [Tesla Investor Shareholder Reports](https://ir.tesla.com/#quarterly-disclosure)"
        )
        filepath = self._upload_doc()
        logger.debug(f"Loaded Filepath: {filepath}")

        if filepath is not None:
            st.sidebar.info(
                "For None-GPU machine it'll take longer time to extract PDF."
            )

            element_list = (
                self._read_pdf(
                    filepath=filepath,
                    extract_images_in_pdf=True,
                    infer_table_structure=True,
                )
                if "element_list" not in st.session_state
                else st.session_state["element_list"]
            )
            st.session_state["element_list"] = element_list
            with st.sidebar.expander("Element types"):
                st.write(set([type(element) for element in element_list]))

            if "retriever" not in st.session_state:
                logger.debug("Creating retriever")
                doc_elements = SummaryCreator().create(element_list)
                doc_summaries = SummaryChain().execute(doc_elements)
                retriever = RetrieverBuilder().build(doc_elements, doc_summaries)
                st.session_state["retriever"] = retriever
                with st.expander("Doc elements"):
                    st.write("Text elements")
                    st.write(doc_elements.text_doc_elements)
                    st.write("Table elements")
                    st.write(doc_elements.table_doc_elements)
                with st.expander("Doc summaries"):
                    st.write("Text summaries")
                    st.write(doc_summaries.text_doc_summaries)
                    st.write("Table summaries")
                    st.write(doc_summaries.table_doc_summaries)

        query = st.text_input("Query")
        if (
            query is not None
            and len(query.strip()) > 0
            and "retriever" in st.session_state
        ):
            logger.debug(f"Query: {query}")
            retriever = st.session_state["retriever"]
            st.write(QueryChain(retriever).invoke(query.strip()))


if __name__ == "__main__":
    App()()

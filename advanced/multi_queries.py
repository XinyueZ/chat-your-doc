import asyncio
import os
from pathlib import Path
from typing import Any, List, Tuple

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain.vectorstores import Chroma
from langchain.vectorstores.chroma import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from llama_index.core import (
    PromptTemplate,
    QueryBundle,
    ServiceContext,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.embeddings.utils import EmbedType
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.indices.base import BaseIndex
from llama_index.core.llms.utils import LLMType
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import BaseQueryEngine, RetrieverQueryEngine
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.service_context import ServiceContext
from llama_index.llms.openai import OpenAI
from loguru import logger
from pydantic import FilePath
from regex import F
from rich.pretty import pprint
from tqdm.asyncio import tqdm

import logging
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

K = 5

VERBOSE = True


def pretty_print(title: str = None, content: Any = None):
    if not VERBOSE:
        return

    if title is None:
        print(content)
        return
    print(title)
    pprint(content)


class MultiQueriesRetriever(BaseRetriever):
    def __init__(self, base_retriever: BaseRetriever, model: OpenAI):
        self.template = PromptTemplate(
            """You are an AI language model assistant. Your task is to generate Five
    different versions of the given user question to retrieve relevant documents from a vector
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search.
    Provide these alternative questions seperated by newlines only.

    For example, these alternative questions:

    'What is Bill Gates known for?'
│   "Can you provide information about Bill Gates' background?"

    Not:

    '1. What is Bill Gates known for?'
│   "2. Can you provide information about Bill Gates' background?"

    Original question: {question}"""
        )
        self._retrievers = [base_retriever]
        self.base_retriever = base_retriever
        self.model = model

    @classmethod
    def flatten(cls, lst: List[List[Any]]) -> List[Any]:
        return [element for sublist in lst for element in sublist]

    def gen_queries(self, query: str) -> List[str]:
        gen_queries_model = OpenAI(model="gpt-3.5-turbo-0125", temperature=1.5)
        prompt = self.template.format(question=query)
        res = gen_queries_model.complete(prompt)
        return res.text.split("\n")

    async def run_gen_queries(
        self, generated_queries: List[str]
    ) -> List[NodeWithScore]:
        tasks = list(map(lambda q: self.base_retriever.aretrieve(q), generated_queries))
        res = await tqdm.gather(*tasks)
        return MultiQueriesRetriever.flatten(res)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return list()

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query: str = query_bundle.query_str
        generated_queries: List[str] = self.gen_queries(query)
        pretty_print("generated_queries", generated_queries)
        node_with_scores = await self.run_gen_queries(generated_queries)
        node_with_scores_uniqued = dict()
        # Simplely removing duplicated nodes in this notebook.
        # For Fusion with ranking, ref:https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
        node_with_scores_uniqued = {
            node_with_score.get_content(): node_with_score
            for node_with_score in node_with_scores
        }
        return node_with_scores_uniqued.values()


class MultiQueriers:
    def __init__(
        self,
        base_retriever: BaseRetriever,
        base_query_engine: BaseQueryEngine,
        model: OpenAI,
        sub_queries_in_bundle_to_answer: bool = True,
    ):
        self.base_retriever = base_retriever
        self.base_query_engine = base_query_engine
        self.model = model
        self.sub_queries_in_bundle_to_answer = sub_queries_in_bundle_to_answer
        self.gen_q_template = PromptTemplate(
            """You are an AI language model assistant. Your task is to generate Five
    different versions of the given user question to retrieve relevant documents from a vector
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search.
    Provide these alternative questions seperated by newlines only.

    For example, these alternative questions:

    'What is Bill Gates known for?'
│   "Can you provide information about Bill Gates' background?"

    Not:

    '1. What is Bill Gates known for?'
│   "2. Can you provide information about Bill Gates' background?"

    Original question: {question}"""
        )
        self.qa_prompt_template = PromptTemplate(
            """Here is the question you need to answer:

    \n --- \n {query_str} \n --- \n

    Here is any available background question + answer pairs:

    \n --- \n {q_a_pairs} \n --- \n

    Here is additional context relevant to the question:

    \n --- \n {context_str} \n --- \n

    Use the above context and any background question + answer pairs to answer the question: \n {query_str}
    """
        )

    def gen_queries(self, query: str) -> List[str]:
        gen_queries_model = OpenAI(model="gpt-3.5-turbo-0125", temperature=1.5)
        prompt = self.gen_q_template.format(question=query)
        res = gen_queries_model.complete(prompt)
        return res.text.split("\n")

    def query_by_retriever(self, query_str: str) -> str:
        nodes = self.base_retriever.retrieve(query_str)
        res = "\n\n".join([n.node.get_content() for n in nodes])
        return res

    async def run_gen_queries(self, generated_queries: List[str]) -> str:
        sub_query_qa_pairs = list()
        if self.sub_queries_in_bundle_to_answer:
            # Answer all queries in one bundle.
            tasks = list(
                map(lambda q: self.base_query_engine.aquery(q), generated_queries)
            )
            res = await tqdm.gather(*tasks)
            for idx, (query, answer) in enumerate(zip(generated_queries, res)):
                qa_pair = f"Question {idx}: {query}\nAnswer: {answer}\n"
                sub_query_qa_pairs.append(qa_pair)
            pretty_print("sub_query_qa_pairs", sub_query_qa_pairs)
            return "\n\n".join(sub_query_qa_pairs)
        else:
            # Answer queries in step down.
            # One sub-query will be answer based on the context of sub-query,
            # history of pairs of previous queries and answers.
            for idx, query in enumerate(generated_queries):
                pretty_print(f"{idx}. query", query)
                pretty_print(f"{idx}. sub_query_qa_pairs", sub_query_qa_pairs)
                context_str = self.query_by_retriever(query)
                sub_query = self.qa_prompt_template.format(
                    query_str=query,
                    q_a_pairs="\n\n".join(sub_query_qa_pairs),
                    context_str=context_str,
                )
                pretty_print("sub_query", sub_query)
                answer: str = self.model.complete(sub_query)
                qa_pair = f"Question {idx}: {query}\nAnswer: {answer}\n"
                sub_query_qa_pairs.append(qa_pair)
            return "\n\n".join(sub_query_qa_pairs)

    def query(self, query_str: str) -> str:
        return ""

    async def aquery(self, query_str: str) -> str:
        generated_queries: List[str] = self.gen_queries(query_str)
        sub_query_qa_pairs: str = await self.run_gen_queries(generated_queries)
        context_str = self.query_by_retriever(query_str)
        final_query: str = self.qa_prompt_template.format(
            query_str=query_str, q_a_pairs=sub_query_qa_pairs, context_str=context_str
        )
        pretty_print("final_query", final_query)
        response: str = self.model.complete(final_query)
        return response.text


class BaseQuerier:
    def __init__(self, **kwargs) -> None:
        logger.debug(f"Querier initialized with {kwargs}")
        self.temperature = kwargs.get("temperature", 1.5)

    async def aquery(self, query_text: str) -> str:
        return f"""{query_text}

Only answer based on the context you have, don't use any external or additional information to makeup the answer.
"""

    def query(self, query_text: str) -> str:
        return f"""{query_text}

Only answer based on the context you have, don't use any external or additional information to makeup the answer.
"""


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
        vector_store = Chroma.from_documents(chunks, embedding=OpenAIEmbeddings())
        self.model = ChatOpenAI(
            temperature=self.temperature,
            model="gpt-4-0125-preview",
        )
        base_retriever = vector_store.as_retriever(search_kwargs={"k": K})
        self.final_retriever = MultiQueryRetriever.from_llm(base_retriever, self.model)

    async def aquery(self, query_text: str) -> str:
        tmpl = """
        You are an assistant to answer a question from user with a context.

        Context:
        {context}

        Question:
        {question}
 
        """
        prompt = ChatPromptTemplate.from_template(tmpl)
        chain = (
            {"question": RunnablePassthrough(), "context": self.final_retriever}
            | prompt
            | self.model
            | StrOutputParser()
        )
        return chain.invoke(query_text)


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
        embs = "default"
        service_context: ServiceContext = self.create_service_context(self.model, embs)
        vector_index: BaseIndex = VectorStoreIndex.from_documents(
            self.docs,
            service_context=service_context,
            show_progress=True,
            transformations=[SentenceSplitter()],
        )
        base_retriever = vector_index.as_retriever(similarity_top_k=K)
        query_engine = vector_index.as_query_engine()

        solution_selector = kwargs.get("solution_selector", 0)
        if solution_selector == 0:
            self.solution = RetrieverQueryEngine(
                MultiQueriesRetriever(base_retriever, self.model)
            )
        elif solution_selector == 1:
            self.solution = MultiQueriers(
                base_retriever,
                query_engine,
                self.model,
                sub_queries_in_bundle_to_answer=True,
            )
        else:
            self.solution = MultiQueriers(
                base_retriever,
                query_engine,
                self.model,
                sub_queries_in_bundle_to_answer=False,
            )

    def create_service_context(self, llm: LLMType, embs: EmbedType) -> ServiceContext:
        return ServiceContext.from_defaults(
            llm=self.model,
            embed_model=embs,
        )

    async def aquery(self, query_text: str) -> str:
        query_text = await super().aquery(query_text)
        return str(await self.solution.aquery(query_text))


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
                    LlamaIndexQuerier(
                        Path(temp_file_path),
                        temperature=temperature,
                        solution_selector=0,
                    ),
                    LlamaIndexQuerier(
                        Path(temp_file_path),
                        temperature=temperature,
                        solution_selector=1,
                    ),
                    LlamaIndexQuerier(
                        Path(temp_file_path),
                        temperature=temperature,
                        solution_selector=2,
                    ),
                )
                return st.session_state["queries"]
        return None


async def main():
    def clear_query_input():
        st.session_state["query_input"] = ""

    st.sidebar.radio(
        "Method",
        [
            "MultiQueryRetriever(LangChain)",
            "MultiQueriesRetriever(Llama-Index, retrieval)",
            "MultiQueriers(Llama-Index, In-Bundle)",
            "MultiQueriers(Llama-Index, Step-down)",
        ],
        index=0,
        key="method_selector",
        on_change=clear_query_input,
    )

    st.sidebar.write(
        "[Github](https://github.com/XinyueZ/chat-your-doc/blob/master/notebooks/multi_queries_retrieval.ipynb)"
    )
    st.sidebar.write(
        "[Notebook](https://colab.research.google.com/drive/1HKv85boODXbU944s3tanL-nBRwin7JAq?usp=sharing)"
    )
    st.sidebar.write("##### Try to play with this doc:")
    st.sidebar.write(
        "[BAIDU, INC. CODE OF BUSINESS CONDUCT AND ETHICS](https://ir.baidu.com/static-files/584e5454-279c-4ffb-8f19-ed64cd054213)"
    )

    temperature: float = st.sidebar.slider(
        "Tempetrature",
        0.0,
        1.8,
        1.0,
        key="temperature_slider",
    )

    queries: Tuple[BaseQuerier] = doc_uploader(temperature=temperature)

    if queries is None:
        return

    query_text = st.text_input(
        "Query",
        key="query_text",
        placeholder="Enter your query here",
    )

    if query_text is not None and query_text != "":
        method_to_querier = {
            "MultiQueryRetriever(LangChain)": queries[0],
            "MultiQueriesRetriever(Llama-Index, retrieval)": queries[1],
            "MultiQueriers(Llama-Index, In-Bundle)": queries[2],
            "MultiQueriers(Llama-Index, Step-down)": queries[3],
        }
        querier = method_to_querier.get(st.session_state["method_selector"], None)
        result: str = await querier.aquery(query_text)
        st.write(result)


if __name__ == "__main__":
    asyncio.run(main())

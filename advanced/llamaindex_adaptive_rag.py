import asyncio
import os
from dataclasses import dataclass
from typing import Any, List, Literal, Dict

import nest_asyncio
from pypdf import mult
import streamlit as st
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
    get_response_synthesizer,
)
from llama_index.core.agent import AgentRunner, FunctionCallingAgentWorker
from llama_index.core.indices.document_summary.base import DocumentSummaryIndex
from llama_index.core.indices.query.query_transform.base import (
    StepDecomposeQueryTransform,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor.metadata_replacement import (
    MetadataReplacementPostProcessor,
)
from llama_index.core.query_engine import (
    BaseQueryEngine,
    CustomQueryEngine,
    RetrieverQueryEngine,
)
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.schema import Document
from llama_index.core.selectors.llm_selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.embeddings.nomic import NomicEmbedding
from llama_index.legacy.postprocessor import CohereRerank, SentenceTransformerRerank
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.cohere import Cohere
from llama_index.llms.gemini import Gemini
from llama_index.llms.groq import Groq
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import MultiStepQueryEngine
from llama_index.legacy.embeddings.langchain import LangchainEmbedding
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from loguru import logger
from rich.pretty import pprint
from tqdm.asyncio import tqdm

nest_asyncio.apply()

VERBOSE = True
WIN_SZ = 3
SIM_TOP_K = 5
RERANK_TOP_K = 3
N_MULTI_STEPS = 5


def pretty_print(title: str = None, content: Any = None):
    if not VERBOSE:
        return

    if title is None:
        print(content)
        return
    print(title)
    pprint(content)


llm_map: Dict[str, LLM] = {
    "anthropic": Anthropic(temperature=0, model="claude-3-haiku-20240307"),
    "openai": OpenAI(temperature=0, model="gpt-4-turbo"),
    "cohere": Cohere(temperature=0, max_tokens=2048),
    "groq": Groq(model="mixtral-8x7b-32768", temperature=0, timeout=60),
}


summary_llm = llm_map[
    st.sidebar.selectbox(
        "Summary LLM", list(llm_map.keys()), index=2, key="summary_llm"
    )
]

multi_step_query_engine_llm = llm_map[
    st.sidebar.selectbox(
        "Multi-step Query Engine LLM",
        list(llm_map.keys()),
        index=3,
        key="multi_step_query_engine_llm",
    )
]
standalone_query_engine_llm = llm_map[
    st.sidebar.selectbox(
        "Standalone Query Engine LLM",
        list(llm_map.keys()),
        index=3,
        key="standalone_query_engine_llm",
    )
]
agent_llm = llm_map[
    st.sidebar.selectbox("Agent LLM", list(llm_map.keys()), index=0, key="agent_llm")
]
chain_llm = llm_map[
    st.sidebar.selectbox("Chain LLM", list(llm_map.keys()), index=0, key="chain_llm")
]
general_llm = llm_map[
    st.sidebar.selectbox(
        "General LLM", list(llm_map.keys()), index=0, key="general_llm"
    )
]
Settings.llm = llm_map[
    st.sidebar.selectbox(
        "Settings LLM", list(llm_map.keys()), index=1, key="settings_llm"
    )
]


Settings.embed_model = LangchainEmbedding(NVIDIAEmbeddings(model="nvolveqa_40k"))
Settings.node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=WIN_SZ,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)


@dataclass
class DataSource:
    name: str
    description: str
    query_engine: BaseQueryEngine
    multi_step_query_engine: BaseQueryEngine

    def __hash__(self):
        return hash((self.name, self.description))

    def __eq__(self, other):
        if isinstance(other, DataSource):
            return self.name == other.name and self.description == other.description
        return False


async def load_docs(file_paths: List[str]) -> List[DataSource]:
    all_doc_src = []
    tasks = [
        index_and_chunks(
            os.path.basename(file_path).split(".")[0],
            SimpleDirectoryReader(input_files=[file_path]).load_data(),
        )
        for file_path in file_paths
    ]
    doc_src_tasks_run = await tqdm.gather(*tasks)
    all_doc_src.extend(doc_src_tasks_run)
    return all_doc_src


async def index_and_chunks(file_name: str, raw_docs: List[Document]) -> DataSource:
    pretty_print("Raw docs", file_name)
    name = file_name
    # check if the name is based on String should match pattern '^[a-zA-Z0-9_-]{1,64}$'
    # required by the LlamaIndex.
    # if not, then replace with a valid name
    if not name.isalnum():
        # replace with a valid name
        name = "file_" + str(hash(name))

    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
        top_n=RERANK_TOP_K, model="BAAI/bge-reranker-base"
    )

    # vector indexing
    retriever = RecursiveRetriever(
        "vector",
        retriever_dict={
            "vector": VectorStoreIndex.from_documents(
                raw_docs, show_progress=True
            ).as_retriever(similarity_top_k=SIM_TOP_K)
        },
        verbose=VERBOSE,
    )

    # summary
    summary = await RetrieverQueryEngine.from_args(
        DocumentSummaryIndex.from_documents(
            raw_docs,
            show_progress=True,
        ).as_retriever(),
        llm=summary_llm,
        response_synthesizer=get_response_synthesizer(
            response_mode=ResponseMode.SIMPLE_SUMMARIZE
        ),
        node_postprocessors=[postproc, rerank],
        verbose=VERBOSE,
    ).aquery("Provide the shortest description of the content.")

    query_engine = RetrieverQueryEngine.from_args(
        retriever,
        llm=standalone_query_engine_llm,
        node_postprocessors=[postproc, rerank],
        verbose=VERBOSE,
    )
    return DataSource(
        name=name,
        description=summary.response,
        query_engine=query_engine,
        multi_step_query_engine=MultiStepQueryEngine(
            query_engine=query_engine,
            query_transform=StepDecomposeQueryTransform(
                llm=multi_step_query_engine_llm, verbose=VERBOSE
            ),
            num_steps=N_MULTI_STEPS,
        ),
    )


def build_mulit_step_query_engine_tools(
    ds_list: List[DataSource],
) -> List[QueryEngineTool]:
    desc_fmt = "Useful for complex queries on the content with multi-step that covers the following dedicated topic:\n{topic}\n"
    return [
        QueryEngineTool(
            query_engine=ds.multi_step_query_engine,
            metadata=ToolMetadata(
                name=ds.name, description=desc_fmt.format(topic=ds.description)
            ),
        )
        for ds in ds_list
    ]


def build_standalone_query_engine_tools(
    ds_list: List[DataSource],
) -> List[QueryEngineTool]:
    desc_fmt = (
        "Useful for simple queries on the content that covers the following dedicated topic:\n{topic}\n"
    )

    return [
        QueryEngineTool(
            query_engine=ds.query_engine,
            metadata=ToolMetadata(
                name=ds.name, description=desc_fmt.format(topic=ds.description)
            ),
        )
        for ds in ds_list
    ]


def build_query_engine_tools_agent_tool(
    query_engine_tools: List[QueryEngineTool],
    base_description: str,
) -> QueryEngineTool:
    agent_worker = FunctionCallingAgentWorker.from_tools(
        query_engine_tools,
        llm=agent_llm,
        verbose=VERBOSE,
        allow_parallel_tool_calls=True,
    )
    agent_runner = AgentRunner(
        agent_worker,
        llm=agent_llm,
        verbose=VERBOSE,
    )

    description_list = [base_description]
    for tools in query_engine_tools:
        meta = tools.metadata
        description_list.append(f"Description of {meta.name}:\n{meta.description}\n")
    description = "\n\n".join(description_list)
    return QueryEngineTool(
        query_engine=agent_runner,
        metadata=ToolMetadata(description=description),
    )


class LLMQueryEngine(CustomQueryEngine):
    """RAG String Query Engine."""

    llm: LLM

    def custom_query(self, query_str: str):
        return str(self.llm.complete(query_str))


def build_fallback_query_engine_tool() -> QueryEngineTool:
    return QueryEngineTool(
        query_engine=LLMQueryEngine(llm=general_llm),
        metadata=ToolMetadata(
            name="General queries as fallback",
            description=(
                "Useful for information about general queries other than specific data sources, as fallback action if no other tool is selected."
            ),
        ),
    )


def build_adaptive_rag_chain(ds_list: List[DataSource]) -> RouterQueryEngine:
    standalone_query_engine_tools = build_standalone_query_engine_tools(ds_list)
    standalone_query_engine_tools_agent_tool = build_query_engine_tools_agent_tool(
        build_standalone_query_engine_tools(ds_list),
        "Useful for queries that span multiple and cross-docs, the docs should cover different topics:\n",
    )

    multi_step_query_engine_tools = build_mulit_step_query_engine_tools(ds_list)
    multi_step_query_engine_tools_agent_tool = build_query_engine_tools_agent_tool(
        build_mulit_step_query_engine_tools(ds_list),
        "Useful for complex queries that span multiple and cross-docs with the help of multi-step, the docs should cover different topics:\n",
    )

    fallback_query_engine_tool = build_fallback_query_engine_tool()
    query_engine_tools = (
        multi_step_query_engine_tools
        + [multi_step_query_engine_tools_agent_tool]
        + standalone_query_engine_tools
        + [standalone_query_engine_tools_agent_tool]
        + [fallback_query_engine_tool]
    )
    return RouterQueryEngine.from_defaults(
        llm=chain_llm,
        selector=LLMSingleSelector.from_defaults(llm=chain_llm),
        query_engine_tools=query_engine_tools,
        verbose=VERBOSE,
    )


async def doc_uploader() -> BaseQueryEngine:
    with st.sidebar:
        uploaded_docs = st.file_uploader(
            "# Upload one text content file",
            key="doc_uploader",
            accept_multiple_files=True,
        )
        if not uploaded_docs:
            st.session_state["file_name"] = None
            st.session_state["queries"] = None
            logger.debug("No file uploaded")
            return None
        if uploaded_docs:
            pretty_print("Uploaded files", uploaded_docs)
            tmp_dir = "tmp/"
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir)
            file_names = []
            for uploaded_doc in uploaded_docs:
                temp_file_path = os.path.join(tmp_dir, f"{uploaded_doc.name}")
                with open(temp_file_path, "wb") as file:
                    file.write(uploaded_doc.getvalue())
                    file_name = uploaded_doc.name
                    logger.debug(f"Uploaded {file_name}")
                    uploaded_doc.flush()
                    uploaded_doc.close()
                file_names.append(temp_file_path)
            all_same_files = (
                all(
                    [
                        file_name == st.session_state["file_names"][idx]
                        for idx, file_name in enumerate(file_names)
                    ]
                )
                if st.session_state.get("file_names")
                else False
            )
            if all_same_files:
                logger.debug("Same file, same quiries, no indexing needed")
                return st.session_state["query_engine"]

            logger.debug("New files, new queries, indexing needed")
            st.session_state["file_names"] = file_names
            pretty_print("File names", st.session_state["file_names"])
            with st.spinner("Indexing, it take while depending on the system..."):
                ds_list = await load_docs(st.session_state["file_names"])
            pretty_print("Data sources", ds_list)
            st.session_state["query_engine"] = build_adaptive_rag_chain(ds_list)
            return st.session_state["query_engine"]
        return None


async def main():
    st.sidebar.title("Upload file")
    query_engine = await doc_uploader()
    if query_engine is None:
        pretty_print("Has query_engine", "No query_engine")
        return
    query_text = st.text_input(
        "Query",
        key="query_text",
        placeholder="Enter your query here",
    ).strip()
    if query_text is not None and query_text != "":
        final_res = await query_engine.aquery(query_text)
        st.write(str(final_res))


if __name__ == "__main__":
    asyncio.run(main())

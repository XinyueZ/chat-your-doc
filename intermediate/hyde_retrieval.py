import asyncio
import os
from typing import Any, List, Tuple

import streamlit as st
from llama_index.core import (
    PromptTemplate,
    QueryBundle,
    ServiceContext,
    SimpleDirectoryReader,
    VectorStoreIndex,
    get_response_synthesizer,
)
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.embeddings.utils import EmbedType
from llama_index.core.indices.base import BaseIndex
from llama_index.core.indices.postprocessor import (
    MetadataReplacementPostProcessor,
    SentenceTransformerRerank,
)
from llama_index.core.llms.utils import LLMType
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers.base import BaseSynthesizer
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.llms.groq import Groq
from llama_index.llms.ollama import Ollama
from loguru import logger
from rich.pretty import pprint
from tqdm.asyncio import tqdm

os.environ["LANGCHAIN_PROJECT"] = "hyde_retrieval"


def pretty_print(title: str = None, content: Any = None):
    if title is None:
        print(content)
        return
    print(title)
    pprint(content)


SIM_TOP_K = 5
RERANK_TOP_K = 5
WIN_SZ = 5

llm_temperature = st.sidebar.slider(
    "LLM temperature for RAG",
    0.0,
    2.0,
    0.0,
    key="llm_temperature_slider",
)
hypo_gen_temperature = st.sidebar.slider(
    "Hypothesis generation LLM temperature",
    0.0,
    2.0,
    1.5,
    key="hypo_gen_temperature_slider",
)

llm: BaseLLM = Groq(model="mixtral-8x7b-32768", temperature=llm_temperature)
hypo_gen_model: BaseLLM = Ollama(
    model="gemma:2b-instruct", temperature=hypo_gen_temperature
)
translation_model: BaseLLM = Ollama(model="gemma:2b-instruct", temperature=0.0)
# llm: BaseLLM = Ollama(model="gemma:2b-instruct", temperature=llm_temperature)
# hypo_gen_model: BaseLLM = Groq(
#     model="mixtral-8x7b-32768", temperature=hypo_gen_temperature
# )
# embs = OpenAIEmbedding(model=OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002)
embs = "local:BAAI/bge-small-en-v1.5"

st.session_state["translate_to_chinese"] = st.sidebar.checkbox(
    "Translate to Chinese", key="translate", value=False
)
# For debugging
st.session_state["hypo_doc"] = None


def create_service_context(llm: LLMType, embs: EmbedType) -> ServiceContext:
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=WIN_SZ,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )

    return ServiceContext.from_defaults(
        node_parser=node_parser,
        llm=llm,
        embed_model=embs,
    )


service_context: ServiceContext = create_service_context(llm, embs)


class HyDERetriever(BaseRetriever):
    def __init__(self, base_retriever: BaseRetriever, hypo_gen_model: BaseLLM):
        self.base_retriever = base_retriever
        self.hypo_gen_model = hypo_gen_model
        self.hypothesis_template = PromptTemplate(
            """Write a hypothesis document about question as you can.

            Only return the document content without any other information, ie. leading text, title text and so on.
            
            Question: {question}

            """
        )

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return []

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_str: str = query_bundle.query_str
        hypo_doc = self.hypo_gen_model.complete(
            self.hypothesis_template.format(question=query_str)
        ).text.strip()

        # For debugging
        st.session_state["hypo_doc"] = hypo_doc

        return await self.base_retriever.aretrieve(hypo_doc)


def create_retriever(service_context: ServiceContext, file_path: str) -> BaseRetriever:
    input_files: List[str] = [file_path]
    docs: SimpleDirectoryReader = SimpleDirectoryReader(
        input_files=input_files,
    ).load_data()

    vector_index: BaseIndex = VectorStoreIndex.from_documents(
        docs,
        service_context=service_context,
        show_progress=True,
    )

    return vector_index.as_retriever()


def create_query_engine(
    service_context: ServiceContext,
    base_retriever: BaseRetriever,
    hypo_gen_model: BaseLLM,
) -> BaseQueryEngine:
    hyde_retriever = HyDERetriever(base_retriever, hypo_gen_model)
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
        hyde_retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[postproc, rerank],
    )


def doc_uploader(service_context: ServiceContext) -> Tuple[BaseRetriever] | None:
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
                    return st.session_state["retriever"]

                logger.debug("New file, new queries, indexing needed")
                st.session_state["file_name"] = file_name
                st.session_state["retriever"] = create_retriever(
                    service_context, temp_file_path
                )
                return st.session_state["retriever"]
        return None


async def main():
    st.sidebar.title("Upload file")
    st.sidebar.write(
        "[Suggest this file to ask some questions](https://dl.dropbox.com/scl/fi/xojn7rk5drda8ba4i90xr/4b1ca7c6-b279-4ed9-961a-484cadf8dd16.pdf?rlkey=aah3wklftddsgw7g5lrkv2tg4&dl=0)"
    )
    retriever: BaseRetriever | None = doc_uploader(service_context)
    if retriever is None:
        return
    query_engine = create_query_engine(service_context, retriever, hypo_gen_model)
    query_text = st.text_input(
        "Query",
        key="query_text",
        placeholder="Enter your query here",
    )

    if query_text is not None and query_text != "":
        final_res: RESPONSE_TYPE = await query_engine.aquery(query_text)
        final_res_str: str = final_res.response
        hypo_doc = st.session_state["hypo_doc"]

        if st.session_state.get("translate_to_chinese", False):
            tasks = [
                translation_model.acomplete(f"Translate the follwing text into Chinese:\n{final_res_str}"),
                translation_model.acomplete(f"Translate the follwing text into Chinese:\n{hypo_doc}"),
            ]
            res = await tqdm.gather(*tasks)
            final_res_str, hypo_doc = res[0].text.strip(), res[1].text.strip()

        st.write("### Final result")
        st.write(final_res_str)

        with st.expander("###  Generated document"):
            st.write(hypo_doc)


if __name__ == "__main__":
    asyncio.run(main())
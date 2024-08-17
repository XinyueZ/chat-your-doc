# Description: (LangChain) Retrieval Augmented Generation with Citations
# Inspired by
# https://zilliz.com/blog/retrieval-augmented-generation-with-citations
# with Llama-Index version:
# 1. https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/query_engine/citation_query_engine.ipynb
# 2. https://www.linkedin.com/posts/llamaindex_check-out-our-new-video-from-ravi-theja-desetty-activity-7229883929594372096-IhhC
#
# Different:
# 1. Use langchain to build the citation chain
# 2. Use two transformers to split the source document and citation chunking.
# 3. Reranking is driven by the GPT model.

# %%
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, cast

from icecream import ic
from langchain import hub
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import UnstructuredURLLoader, WebBaseLoader
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import BaseDocumentTransformer, Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_mistralai import MistralAIEmbeddings
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters.sentence_transformers import (
    SentenceTransformersTokenTextSplitter,
)
from loguru import logger
from rich.pretty import pprint as pp
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors.rankllm_rerank import RankLLMRerank


# monitor of langsmith
# os.environ["LANGCHAIN_PROJECT"] = "langchain_citation_chain"

# %%
MAX_TOKEN = 2048
TEMPERATURE = 0.0
RETRIEVER_K = 10
RERANKING_N = 5
DEFAULT_CITATION_CHUNK_SIZE = 512
DEFAULT_CITATION_CHUNK_OVERLAP = 32

# %%


class ModelModule:
    embed: Embeddings
    llm: BaseChatModel
    compressor: BaseDocumentCompressor


class GoogleModule(ModelModule):
    llm: BaseChatModel = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest", temperature=TEMPERATURE, max_tokens=MAX_TOKEN
    )
    embed: Embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    compressor: BaseDocumentCompressor = RankLLMRerank(
        top_n=RERANKING_N, model="gpt", gpt_model="gpt-4o-mini"
    )


class GroqModule(ModelModule):
    llm: BaseChatModel = ChatGroq(
        model="llama-3.1-70b-versatile", temperature=TEMPERATURE, max_tokens=MAX_TOKEN
    )
    embed: Embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    compressor: BaseDocumentCompressor = RankLLMRerank(
        top_n=RERANKING_N, model="gpt", gpt_model="gpt-4o-mini"
    )


# Some other embedding models:
# OpenAIEmbeddings(model="text-embedding-3-large")
# MistralAIEmbeddings(model="mistral-embed")
# OllamaEmbeddings(model="nomic-embed-text", show_progress=False)
# GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# %%
module_map = {"google": GoogleModule, "groq": GroqModule}

current_module = module_map["groq"]()


retriever_transformer = SemanticChunker(current_module.embed)
chunking_transformer = SentenceTransformersTokenTextSplitter(
    chunk_overlap=DEFAULT_CITATION_CHUNK_OVERLAP,
    chunk_size=DEFAULT_CITATION_CHUNK_SIZE,
)


# %%
@dataclass
class CitationResult:
    result: str
    citations: Sequence[str]


class CitationChain:

    def __init__(
        self,
        current_module: ModelModule,
        retriever_transformer: BaseDocumentTransformer,
        chunking_transformer: BaseDocumentTransformer,
    ):
        """
        Initialize a MyClass object.

        Args:
            current_module (ModelModule): The module of models.
            retriever_transformer (BaseDocumentTransformer): The retriever transformer, for init doc splitting.
            chunking_transformer (BaseDocumentTransformer): The chunking transformer, for citation chunking.
        """
        self.current_module = current_module
        self.retriever_transformer = retriever_transformer
        self.chunking_transformer = chunking_transformer
        self.citation_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "user",
                    (
                        "Please provide an answer based solely on the provided sources. "
                        "When referencing information from a source, "
                        "cite the appropriate source(s) using their corresponding numbers. "
                        "Every answer should include at least one source citation. "
                        "Only cite a source when you are explicitly referencing it if it is in the Context. "
                        "If none of the sources are helpful, you should indicate that. "
                        "For example:\n"
                        "Source 1:\n"
                        "The sky is red in the evening and blue in the morning.\n"
                        "Source 2:\n"
                        "Water is wet when the sky is red.\n"
                        "Query: When is water wet?\n"
                        "Answer: Water will be wet when the sky is red [2], "
                        "which occurs in the evening [1].\n"
                        "[1] means from Source 1, [2] means from Source 2."
                        "You must always display the citation source numbers correctly between the reference location and the source location. "
                        "You don't need to show the list of sources in the answer. "
                        "Now it's your turn. Below are several numbered sources of information:"
                        "\n------\n"
                        "Context: {context}"
                        "\n------\n"
                        "Query: {question}\n"
                        "Answer: "
                    ),
                )
            ]
        )

        self.refine_citation_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "user",
                    (
                        "Please provide an answer based solely on the provided sources. "
                        "When referencing information from a source, "
                        "cite the appropriate source(s) using their corresponding numbers. "
                        "Every answer should include at least one source citation. "
                        "Only cite a source when you are explicitly referencing it if it is in the Context. "
                        "If none of the sources are helpful, you should indicate that. "
                        "For example:\n"
                        "Source 1:\n"
                        "The sky is red in the evening and blue in the morning.\n"
                        "Source 2:\n"
                        "Water is wet when the sky is red.\n"
                        "Query: When is water wet?\n"
                        "Answer: Water will be wet when the sky is red [2], "
                        "which occurs in the evening [1].\n"
                        "[1] means from Source 1, [2] means from Source 2."
                        "You must always display the citation source numbers correctly between the reference location and the source location. "
                        "Now it's your turn. "
                        "We have provided an existing answer: {existing_answer}"
                        "Below are several numbered sources of information. "
                        "Use them to refine the existing answer. "
                        "Check if the citations between reference and source are correct and if the answer is accurate, if the reference is pointed to a unknow source, please remove the citation number.  "
                        "If the provided sources are not helpful, you will repeat the existing answer."
                        "\nBegin refining!"
                        "\n------\n"
                        "Context: {context}"
                        "\n------\n"
                        "Query: {question}\n"
                        "Answer: "
                    ),
                )
            ]
        )
        self.compressor = current_module.compressor

    def load_docs(self, urls: Sequence[str], loader_exe_fn) -> Sequence[Document]:
        """Load the docs from the given Urls."""
        loader = WebBaseLoader(urls)
        docs = loader_exe_fn(loader)
        logger.info(docs)
        return docs

    def retrieve(self, query: str, urls: Sequence[str]) -> Sequence[Document]:
        """Retrieve the docs for the content of the given Urls."""
        docs = self.load_docs(
            urls, lambda loader: loader.load_and_split(self.retriever_transformer)
        )

        db = FAISS.from_documents(docs, self.current_module.embed)
        return db.as_retriever(search_kwargs={"k": RETRIEVER_K}).invoke(query)

    def make_citation_chunks(
        self, retrieved_docs: Sequence[Document]
    ) -> Sequence[Document]:
        """
        Modify retrieved nodes to create granular sources for citations.

        Takes a list of Documents and splits their content
        into smaller chunks, creating new list of Documents for each chunk.
        Each new node is labeled as a numbered source, allowing for more precise
        citation in query results.
        """
        chunks: Sequence[Document] = []
        for retrieved_doc in retrieved_docs:
            meta = retrieved_doc.metadata
            meta_source = meta.get("source", "")
            meta_title = meta.get("title", "")
            meta_language = meta.get("language", "")

            doc2chunks = self.chunking_transformer.split_documents([retrieved_doc])
            for chunk in doc2chunks:
                page_content = f"Source {len(chunks)+1}:\n{chunk.page_content}\n"
                new_chunk = Document(
                    page_content=page_content,
                    metadata={
                        "source": meta_source,
                        "title": meta_title,
                        "language": meta_language,
                    },
                )
                chunks.append(new_chunk)
            del doc2chunks
        return chunks

    def __call__(self, question: str, urls: Sequence[str]) -> CitationResult:
        return self.qa(question, urls)

    def qa(self, question: str, urls: Sequence[str]) -> CitationResult:
        """Q&A for the content of the given Urls."""
        docs = self.retrieve(question, urls)
        logger.info(f"retrieved docs: {len(docs)}")
        chunks = self.make_citation_chunks(docs)
        logger.info(f"citation chunks: {len(chunks)}")

        # Tip: Further implementation for example vector-summary:
        # use a vector database for chunks and a regular database for parent documents;
        # here, just ignore this and only save chunks (vectors).
        db = FAISS.from_documents(chunks, self.current_module.embed)
        base_retriever = db.as_retriever(search_kwargs={"k": RETRIEVER_K})
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor, base_retriever=base_retriever
        )
        context = compression_retriever.invoke(question)
        chain = (
            {
                "question": lambda x: x["question"],
                "context": lambda x: x["context"],
                "existing_answer": self.citation_prompt | self.current_module.llm,
            }
            | self.refine_citation_prompt
            | self.current_module.llm
            | StrOutputParser()
        )
        citations = list(map(lambda chunk: chunk.page_content, context))
        str_res = chain.invoke(
            input={"question": question, "context": "\n".join(citations)}
        )

        del chunks, docs
        return CitationResult(result=str_res, citations=citations)


# %%
def main():
    citation_chain = CitationChain(
        current_module, retriever_transformer, chunking_transformer
    )
    result = citation_chain.qa(
        (
            "An in-depth introduction to Germany, about its culture, economy, and industry. "
            "answered for those without any prior knowledge."
        ),
        [
            "https://en.wikipedia.org/wiki/Germany",
        ],
    )
    ic(result.result)
    pp(result.citations)


# %%
if __name__ == "__main__":
    main()

# %%

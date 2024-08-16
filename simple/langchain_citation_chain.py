# %%
import os
import sys
from typing import (Any, Dict, Iterable, List, Literal, Optional, Sequence,
                    Tuple, cast)

from icecream import ic
from langchain import hub
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import (UnstructuredURLLoader,
                                                  WebBaseLoader)
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import BaseDocumentTransformer, Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import (ChatGoogleGenerativeAI,
                                    GoogleGenerativeAIEmbeddings)
from langchain_groq import ChatGroq
from langchain_mistralai import MistralAIEmbeddings
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_openai import OpenAIEmbeddings
from loguru import logger
from rich.pretty import pprint as pp

# monitor of langsmith
# os.environ["LANGCHAIN_PROJECT"] = "langchain_citation_chain"

# %%
MAX_TOKEN = 2048
TEMPERATURE = 0.0
RETRIEVER_K = 5

# %%


class ModelModule:
    embed: Embeddings
    llm: BaseChatModel


class GoogleModule(ModelModule):
    llm: BaseChatModel = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest", temperature=TEMPERATURE, max_tokens=MAX_TOKEN
    )
    embed: Embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")


class GroqModule(ModelModule):
    llm: BaseChatModel = ChatGroq(
        model="llama-3.1-70b-versatile", temperature=TEMPERATURE, max_tokens=MAX_TOKEN
    )
    embed: Embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


# OpenAIEmbeddings(model="text-embedding-3-large")
# MistralAIEmbeddings(model="mistral-embed")
# OllamaEmbeddings(model="nomic-embed-text", show_progress=False)
# GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# %%
module_map = {"google": GoogleModule, "groq": GroqModule}

current_module = module_map["groq"]()


transformer = SemanticChunker(current_module.embed)

# %%

# prompt = hub.pull("hwchase17/llama-rag")
citation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            (
                "Please provide an answer based solely on the provided sources. "
                "When referencing information from a source, "
                "cite the appropriate source(s) using their corresponding numbers. "
                "Every answer should include at least one source citation. "
                "Only cite a source when you are explicitly referencing it. "
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
                "Now it's your turn. Below are several numbered sources of information:"
                "\n------\n"
                "{context}"
                "\n------\n"
                "Query: {question}\n"
                "Answer: "
            ),
        )
    ]
)

refine_citation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            (
                "Please provide an answer based solely on the provided sources. "
                "When referencing information from a source, "
                "cite the appropriate source(s) using their corresponding numbers. "
                "Every answer should include at least one source citation. "
                "Only cite a source when you are explicitly referencing it. "
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
                "Now it's your turn. "
                "We have provided an existing answer: {existing_answer}"
                "Below are several numbered sources of information. "
                "Use them to refine the existing answer. "
                "If the provided sources are not helpful, you will repeat the existing answer."
                "\nBegin refining!"
                "\n------\n"
                "{context}"
                "\n------\n"
                "Query: {question}\n"
                "Answer: "
            ),
        )
    ]
)


# %%


def load_docs(urls: Sequence[str], loader_exe_fn) -> Sequence[Document]:
    """Load the docs from the given Urls."""
    loader = WebBaseLoader(urls)
    docs = loader_exe_fn(loader)
    logger.info(docs)
    return docs


def retrieve(query: str, urls: Sequence[str]) -> Sequence[Document]:
    """Retrieve the docs for the content of the given Urls."""
    docs = load_docs(urls, lambda loader: loader.load_and_split(transformer))

    db = FAISS.from_documents(docs, current_module.embed)
    return db.as_retriever(search_kwargs={"k": RETRIEVER_K}).invoke(query)


# %%
# retrieval_res = retrieve(
#     "Some briefings to introduce the city", ["https://en.wikipedia.org/wiki/France"]
# )
# pp(retrieval_res)


# %%
def make_citation_chunks(retrieved_docs: Sequence[Document]) -> Sequence[Document]:
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

        doc2chunks = transformer.transform_documents([retrieved_doc])
        for chunk in doc2chunks:
            page_content = f"Source {len(chunks)+1}:\n{meta_title}\n{chunk.page_content}\n{meta_source}\n"
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


def qa(question: str, urls: Sequence[str]) -> str:
    """Q&A for the content of the given Urls."""
    docs = retrieve(question, urls)
    logger.info(f"retrieved docs: {len(docs)}")
    chunks = make_citation_chunks(docs)
    logger.info(f"citation chunks: {len(chunks)}")
    pp(chunks)
    # Tip: Further implementation for example vector-summary:
    # use a vector database for chunks and a regular database for parent documents;
    # here, just ignore this and only save chunks (vectors).
    db = FAISS.from_documents(chunks, current_module.embed)

    chain = (
        {
            "question": lambda x: x["question"],
            "context": lambda x: x["context"],
            "existing_answer": citation_prompt | current_module.llm,
        }
        | refine_citation_prompt
        | current_module.llm
        | StrOutputParser()
    )
    return chain.invoke(
        input={
            "question": question,
            "context": db.as_retriever(search_kwargs={"k": RETRIEVER_K}).invoke(
                question
            ),
        }
    )


# %%
qa_res = qa(
    (
        "An in-depth introduction to the German capital city, "
        "answered for those without any prior knowledge."
    ),
    ["https://en.wikipedia.org/wiki/Germany", "https://en.wikipedia.org/wiki/Berlin"],
)
ic(qa_res)
ic("")

# %%

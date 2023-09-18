import os

import streamlit as st
from dotenv import find_dotenv, load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from loguru import logger

_ = load_dotenv(find_dotenv())

# os.getenv('OPENAI_API_KEY')


class App:
    _chain: ConversationalRetrievalChain

    _embeddings: Embeddings

    _persist_directory: str

    def __init__(self) -> None:
        self._persist_directory = "db"

        self._embeddings = (
            OpenAIEmbeddings()
            if "embeddings" not in st.session_state
            else st.session_state["embeddings"]
        )

        llm = (
            ChatOpenAI(model="gpt-4-0613")
            if "llm" not in st.session_state
            else st.session_state["llm"]
        )

        retriever = Chroma(
            persist_directory=self._persist_directory,
            embedding_function=self._embeddings,
        ).as_retriever(
            search_type="similarity"  # or mmr
        )
        # retriever = ContextualCompressionRetriever(
        #     base_compressor=LLMChainExtractor.from_llm(llm),
        #     base_retriever=retriever,
        # )

        if "chain" not in st.session_state:
            logger.debug("Creating new chain")

            self._chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                chain_type="stuff",
                memory=ConversationBufferMemory(
                    input_key="question",
                    output_key="answer",
                    return_messages=True,  # then buffer === buffer_as_messages (a list) instead pure str returning
                ),
                retriever=retriever,
                return_source_documents=True,
                return_generated_question=True,
            )
            st.session_state["chain"] = self._chain
            st.session_state["embeddings"] = self._embeddings
        else:
            logger.debug("Loading existing chain")
            self._embeddings = st.session_state["embeddings"]
            self._chain = st.session_state["chain"]
            self._chain.retriever = retriever

    def _abbr(self, msg) -> str:
        if isinstance(msg, HumanMessage):
            return "user"
        elif isinstance(msg, AIMessage):
            return "assistant"
        else:
            raise ValueError(f"Unknown msg type: {msg}")

    def _load_and_split(self, path: str) -> list[Document]:
        logger.debug(f"Loading {path}")
        loader = PyPDFLoader(path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(docs)

        return texts

    def _store_file(self) -> None:
        uploaded_pdf = st.file_uploader("Upload a PDF")
        if uploaded_pdf:
            temp_file = f"./{uploaded_pdf.name}.pdf"
            with open(temp_file, "wb") as file:
                file.write(uploaded_pdf.getvalue())
                file_name = uploaded_pdf.name
                logger.info(f"Uploaded {file_name}")

            pdf_content = self._load_and_split(path=temp_file)
            vectordb = Chroma.from_texts(
                texts=[c.page_content for c in pdf_content],
                embedding=self._embeddings,
                persist_directory=self._persist_directory,
            )
            vectordb.persist()

            os.remove(temp_file)
            uploaded_pdf = None

    def run(self) -> None:
        st.title("Chat with Your Documents")

        self._store_file()

        st.chat_message(name="ai").write(
            "Hey, I can read your uploaded documents and assist you to understand them."
        )

        for msg in self._chain.memory.buffer:
            st.chat_message(name=self._abbr(msg)).write(msg.content)

        # logger.debug(self._chain.memory.buffer)

        if prompt := st.chat_input(placeholder="Ask questions"):
            st.chat_message(name="user").write(prompt)

            result = self._chain(
                {
                    "question": prompt,
                    "chat_history": self._chain.memory.buffer,
                }
            )

            # logger.debug(
            #     f"""generated_question: {result["generated_question"]}, source_documents: {result["source_documents"]}, answer: {result['answer']}"""
            # )

            st.experimental_rerun()


if __name__ == "__main__":
    App().run()

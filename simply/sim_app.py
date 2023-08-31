import os

import streamlit as st
from dotenv import find_dotenv, load_dotenv
from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from loguru import logger

_ = load_dotenv(find_dotenv())

# os.getenv('OPENAI_API_KEY')


class App:
    def __init__(self) -> None:
        self.persist_directory = "db"
        self.embeddings = OpenAIEmbeddings()

    def load_and_split(self, path: str, is_unstructured: bool = True):
        logger.info(f"Loading {path}, Unstructured: {is_unstructured}")
        loader = UnstructuredPDFLoader(path) if is_unstructured else PyPDFLoader(path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(docs)

        return texts

    def __call__(self):
        pdf_options = ["Unstructured PDF", "Normal PDF"]
        pdf_op = st.radio("Select PDF type", pdf_options, index=0)

        uploaded_pdf = st.file_uploader("Upload a PDF")
        if uploaded_pdf:
            temp_file = "./temp.pdf"
            with open(temp_file, "wb") as file:
                file.write(uploaded_pdf.getvalue())
                file_name = uploaded_pdf.name
                logger.info(f"Uploaded {file_name}")

            pdf_content = self.load_and_split(
                path=temp_file, is_unstructured=pdf_op == pdf_options[0]
            )
            vectordb = Chroma.from_texts(
                texts=[c.page_content for c in pdf_content],
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
            )
            vectordb.persist()

            os.remove(temp_file)
            uploaded_pdf = None

            question = st.text_input("Ask a question")
            if question:
                sim_methods = ["similarity_search", "max_marginal_relevance_search"]
                sim_method = st.radio("Similarity method", sim_methods, index=0)
                logger.info(f"Similarity method: {sim_method}")

                m = (
                    vectordb.similarity_search
                    if sim_method == sim_methods[0]
                    else vectordb.max_marginal_relevance_search
                )
                answers = m(
                    question,
                    k=2,
                    fetch_k=6,
                    lambda_mult=1,
                )
                a = st.selectbox("Answers", [a for a in range(len(answers))])
                st.write(f"{answers[int(a)].page_content}")


if __name__ == "__main__":
    # Run:  streamlit run simply/sim_app.py --server.port 8888 --server.enableCORS false
    app = App()
    app()

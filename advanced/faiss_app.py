import pickle

import requests
import streamlit as st
from bs4 import BeautifulSoup
from langchain.chains import VectorDBQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from loguru import logger


class App:
    _llm: ChatOpenAI
    _embeddings: OpenAIEmbeddings

    def __init__(self) -> None:
        self._llm = ChatOpenAI(model="gpt-4-0613")
        self._embeddings = OpenAIEmbeddings()

        self.prompt = """What is {txt}?"""

    def _get_html_text(self, url):
        r = requests.get(url)
        soup = BeautifulSoup(r.text, "html.parser")
        txt = soup.get_text()
        lines = (line.strip() for line in txt.splitlines())
        return "\n".join(line for line in lines if line)

    def __call__(self):
        st.write("# Ask html page")

        url = st.text_input("URL", "https://arxiv.org/abs/2308.14963")
        if url:
            txt = self._get_html_text(url)
            logger.info(txt)

            #
            # Save to vector storage
            # Note:
            #  metadata is a dictionary containing the source of the text, otherwise there will be errors
            #  "ValueError: Document prompt requires documents to have metadata variables: ['source']. Received document with missing metadata: ['source'].""
            #
            faiss_store = FAISS.from_texts(
                [txt], self._embeddings, metadatas=[{"source": url}]
            )
            with open("db/faiss_store.pkl", "wb") as f:
                pickle.dump(faiss_store, f)

            #
            # Read from vector storage and finish query on row
            #
            with open("db/faiss_store.pkl", "rb") as f:
                faiss_store = pickle.load(f)

            #
            # AI -> find out similar docs
            # AI -> answer question
            #
            chain = VectorDBQAWithSourcesChain.from_llm(
                llm=self._llm,
                vectorstore=faiss_store,
            )
            q = st.text_input("What is ...?")
            if q is None or q == "":
                return
            res = chain({"question": self.prompt.format(txt=q)})
            answer = res["answer"]
            logger.debug(answer)
            st.write(answer)


if __name__ == "__main__":
    app = App()
    app()

from os import sep
import streamlit as st
from bs4 import BeautifulSoup
import requests
from loguru import logger
from langchain.text_splitter import CharacterTextSplitter


class App:
    def _get_html_text(self, url):
        r = requests.get(url)
        soup = BeautifulSoup(r.text, "html.parser")
        txt = soup.get_text()
        lines = (line.strip() for line in txt.splitlines())
        return "\n".join(line for line in lines if line)

    def __call__(self):
        st.write("# Load HTML and split into chunks")
        url = st.text_input("URL")
        if url:
            txt = self._get_html_text(url)
            logger.info(txt)

            chunk_size = st.slider("Chunk size", 1, 300, 201)
            splitter = CharacterTextSplitter(chunk_size=chunk_size, separator="\n")
            chunks = splitter.split_text(txt)

            st.write(f"Number of chunks: {len(chunks)}")
            st.write("First 5 chunks:")
            for chunk in chunks[:5]:
                st.write(chunk)


if __name__ == "__main__":
    app = App()
    app()

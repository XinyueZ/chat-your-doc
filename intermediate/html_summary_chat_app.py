from os import sep
import streamlit as st
from bs4 import BeautifulSoup
import requests
from loguru import logger

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate


class App:
    _chat: ChatOpenAI
    _chat_template: ChatPromptTemplate

    def __init__(self) -> None:
        self._chat = ChatOpenAI(model="gpt-4-0613")
        self._chat_template = ChatPromptTemplate.from_messages(
            [
                ("system", "You are reading a text and want to summarize it."),
                ("human", "Your text: {text}"),
            ]
        )

    def _get_html_text(self, url):
        r = requests.get(url)
        soup = BeautifulSoup(r.text, "html.parser")
        txt = soup.get_text()
        lines = (line.strip() for line in txt.splitlines())
        return "\n".join(line for line in lines if line)

    def __call__(self):
        st.write("# Load HTML, summarize the content")
        url = st.text_input("URL", "https://arxiv.org/abs/2308.14963")
        if url:
            txt = self._get_html_text(url)
            logger.info(txt)

            msg = self._chat_template.format_messages(text=txt)

            res = self._chat(msg)
            st.write(res.content)


if __name__ == "__main__":
    app = App()
    app()

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
        sys_tmpl = """Summarize a text by identifying its keypoints. Each keypoint should be represented by:
        keypoint: a sentence of the representation of the keypoint
        keyword: at least three words that represent the keypoint
        Answer only in the format of json that contains a list of keypoint objects."""
        self._chat_template = ChatPromptTemplate.from_messages(
            [
                ("system", sys_tmpl),
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
        st.write(
            "# Load HTML, find keypoints, format in json (plain text)"
        )
        url = st.text_input("URL", "https://arxiv.org/abs/2308.14963")
        if url:
            txt = self._get_html_text(url)
            logger.info(txt)

            msg = self._chat_template.format_messages(text=txt)

            res = self._chat(msg)
            st.code(res.content, language="json")


if __name__ == "__main__":
    app = App()
    app()

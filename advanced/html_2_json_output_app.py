import requests
import streamlit as st
from bs4 import BeautifulSoup
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from loguru import logger


st.set_page_config(
    page_icon=":smiley:",
    layout="wide",
)


class App:
    _chat: ChatOpenAI
    _sys_template: SystemMessagePromptTemplate
    _user_template: HumanMessagePromptTemplate
    _chat_template: ChatPromptTemplate
    _structured_output_parser: StructuredOutputParser
    _lang_list: list[str]

    def __init__(self) -> None:
        self._lang_list = [
            "English",
            "German",
            "Simplified Chinese",
            "Traditional Chinese",
        ]

        self._chat = ChatOpenAI(model="gpt-4-0613", verbose=True)
        sys_tmpl = """Summarize a text by identifying its keypoints. Each keypoint should be represented by:
        keypoint: a sentence of the representation of the keypoint
        keywords: at least three words that represent the keypoint
        Answer only in the format of json that contains a list of keypoint objects.

        You must translate language from {src_lang} to {dist_lang} for the keypoint and keywords. Optimize the result is {optimising}.
        The format instructions is {format_instructions}"""

        self._sys_template = SystemMessagePromptTemplate.from_template(sys_tmpl)
        self._user_template = HumanMessagePromptTemplate.from_template(
            "Your text: {text}"
        )
        self._chat_template = ChatPromptTemplate.from_messages(
            [self._sys_template, self._user_template]
        )

        self._structured_output_parser = StructuredOutputParser.from_response_schemas(
            [
                ResponseSchema(
                    name="summary",
                    description="""array of objects: [
    { "keypoint": string // a sentence of the representation of the keypoint',  "keywords": string // at least three words that represent the keypoint' }
]
""",
                ),
            ]
        )

    def _get_html_text(self, url):
        r = requests.get(url)
        soup = BeautifulSoup(r.text, "html.parser")
        txt = soup.get_text()
        lines = (line.strip() for line in txt.splitlines())
        return "\n".join(line for line in lines if line)

    def _generate_html_page(self, list_output):
        table_html = "<table style='color: white; border: 8px solid white; font-size: 16px; line-height: 1.5em; margin-top: 20px;'><tr><th>Keypoint</th><th>Keywords</th></tr>"
        for i, item in enumerate(list_output):
            if i % 2 == 0:
                row_style = "background-color: gray;"
            else:
                row_style = "background-color: darkgray;"
            table_html += f"<tr style='{row_style}'><td>{item['keypoint']}</td><td>{item['keywords']}</td></tr>"
        table_html += "</table>"
        return f"<html><body>{table_html}</body></html>"

    def __call__(self):
        st.write("# Load HTML, find keypoints, format in json")
        url = st.text_input("URL", "https://arxiv.org/abs/2308.14963")
        if url:
            txt = self._get_html_text(url)
            logger.debug(txt)

            col1, col2 = st.columns(2)
            with col1:
                src_lang = st.selectbox("Source Language", self._lang_list)
            with col2:
                dist_lang = st.selectbox("Destination Language", self._lang_list)
            is_opt = st.checkbox("Optimize result", value=False)

            optimising = "necessary" if is_opt else "unnecessary"

            msg = self._chat_template.format_messages(
                text=txt,
                src_lang=src_lang,
                dist_lang=dist_lang,
                optimising=optimising,
                format_instructions=self._structured_output_parser.get_format_instructions(),
            )

            res = self._chat(msg)
            logger.debug(res.content)

            structured_output = self._structured_output_parser.parse(res.content)
            logger.debug(structured_output)

            list_output = structured_output["summary"]
            logger.debug(list_output[0])

            # Generate HTML page for list_output
            html_page = self._generate_html_page(list_output)

            # Display HTML page in Streamlit
            st.components.v1.html(html_page, height=5200)


if __name__ == "__main__":
    app = App()
    app()

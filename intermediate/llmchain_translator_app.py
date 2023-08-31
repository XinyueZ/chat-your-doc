import streamlit as st
from dotenv import find_dotenv, load_dotenv
from langchain import LLMChain, PromptTemplate
from langchain.llms import OpenAI
from loguru import logger


from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain

_ = load_dotenv(find_dotenv())

# os.getenv('OPENAI_API_KEY')


class App:
    def __init__(self) -> None:
        self.lang_list = [
            "English",
            "German",
            "Simplified Chinese",
            "Traditional Chinese",
        ]

        self.sys_msg_tmpl = SystemMessagePromptTemplate.from_template(
            "You're a machine language translator. You can translate language from {src_lang} to {dist_lang}. Optimize the result is {optimising}."
        )
        self.human_msg_tmpl = HumanMessagePromptTemplate.from_template(
            "Translate this sentence: {sentence}"
        )

        self.chat_prompt = ChatPromptTemplate.from_messages(
            [self.sys_msg_tmpl, self.human_msg_tmpl]
        )

        self.llm_chain = LLMChain(llm=ChatOpenAI(), prompt=self.chat_prompt)

    def __call__(self):
        st.write("# Translator")
        text = st.text_input("Your text")

        col1, col2 = st.columns(2)
        with col1:
            src_lang = st.selectbox("Source Language", self.lang_list)
        with col2:
            dist_lang = st.selectbox("Destination Language", self.lang_list)
        is_opt = st.checkbox("Optimize result", value=False)

        optimising = "necessary" if is_opt else "unnecessary"
        if text:
            answer = self.llm_chain.run(
                src_lang=src_lang,
                dist_lang=dist_lang,
                optimising=optimising,
                sentence=text,
            )
            logger.info(f"Answer: {answer}")
            st.write(answer)


if __name__ == "__main__":
    app = App()
    app()

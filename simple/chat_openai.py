import streamlit as st
from langchain.chat_models import ChatOpenAI
from loguru import logger
from langchain.schema.messages import HumanMessage


class App:
    def __call__(self):
        st.write("# Chat")

        model = st.selectbox(
            "Model",
            [
                "gpt-3.5-turbo-0613",
                "gpt-4-0613",
                "gpt-4-32k-0613",
            ],
        )

        chat = ChatOpenAI(model=model)

        text = st.text_input("Your text")
        if text:
            try:
                answer = chat([HumanMessage(content=text)])
                logger.info(f"Answer: {answer}")
                st.write(answer.content)
            except Exception as e:
                st.error(e)
                logger.error(e)


if __name__ == "__main__":
    app = App()
    app()

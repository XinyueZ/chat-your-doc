import streamlit as st
from dotenv import find_dotenv, load_dotenv
from langchain import LLMChain, PromptTemplate
from langchain.llms import OpenAI
from loguru import logger

_ = load_dotenv(find_dotenv())

# os.getenv('OPENAI_API_KEY')


class App:
    def __init__(self) -> None:
        self.template = """Question: {question}

        Answer: Let's think step by step."""

        self.prompt = PromptTemplate(
            template=self.template, input_variables=["question"]
        )
        self.llm = OpenAI(temperature=0.5)
        self.llm_chain = LLMChain(prompt=self.prompt, llm=self.llm)

    def __call__(self):
        st.write("# Q&A")
        question = st.text_input("Ask a question")
        if question:
            answers = self.llm_chain.run(question)
            logger.info(f"Answers: {answers}")
            st.write(answers)


if __name__ == "__main__":
    app = App()
    app()

import streamlit as st
from langchain.chat_models import ChatOpenAI
from loguru import logger
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.docarray import DocArrayInMemorySearch
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableMap


class App:
    def __init__(self) -> None:
        st.session_state["memory"] = (
            list() if "memory" not in st.session_state else st.session_state["memory"]
        )
        self.embeddings = OpenAIEmbeddings()
        self.model = ChatOpenAI(model="gpt-4-1106-preview")
        self.template = """Answer the question based only on the following context:
            {context}

            Question: {question}

            Notics: Only returns the answer.
        """
        self.prompt = ChatPromptTemplate.from_template(self.template)
        self.output_parser = StrOutputParser()

    def __call__(self):
        mem_txt = st.sidebar.text_input("Memory(press enter)")
        if mem_txt:
            if not st.session_state["memory"].count(mem_txt):
                st.session_state["memory"].append(mem_txt)
            for mem in st.session_state["memory"]:
                st.sidebar.write(mem)
        st.write("# Ask memory")
        question_txt = st.text_input("Ask memory(press enter)")
        if question_txt:
            vectorstore = DocArrayInMemorySearch.from_texts(
                st.session_state["memory"],
                embedding=self.embeddings,
            )
            retriever = vectorstore.as_retriever()
            chain = (
                RunnableMap(
                    {
                        "context": lambda inputs: retriever.get_relevant_documents(
                            inputs["question"]
                        ),
                        "question": lambda inputs: inputs["question"],
                    }
                )
                | self.prompt
                | self.model
                | self.output_parser
            )
            res = chain.invoke({"question": question_txt})
            logger.debug(retriever.get_relevant_documents(question_txt))
            st.write(res)


if __name__ == "__main__":
    app = App()
    app()

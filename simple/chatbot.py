import streamlit as st
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

from loguru import logger

from langchain.schema.messages import HumanMessage, AIMessage


class ChatBot:
    _chain: ConversationChain

    def __init__(self):
        if "chain" not in st.session_state:
            llm = ChatOpenAI(model="gpt-4-0613")

            memory = ConversationBufferMemory(
                return_messages=True  # then buffer === buffer_as_messages (a list) instead pure str returning
            )
            self._chain = ConversationChain(
                llm=llm,
                memory=memory,
                verbose=False,
            )
            st.session_state["chain"] = self._chain
        else:
            self._chain = st.session_state["chain"]

    def _abbr(self, msg):
        if isinstance(msg, HumanMessage):
            return "user"
        elif isinstance(msg, AIMessage):
            return "assistant"
        else:
            raise ValueError(f"Unknown msg type: {msg}")

    def run(self):
        st.title("MyGPT")
        st.chat_message(name="ai").write("Hey, I am your assistant, ask me anything!")

        logger.debug(self._chain.memory.buffer)
        for msg in self._chain.memory.buffer:
            st.chat_message(name=self._abbr(msg)).write(msg.content)
        if prompt := st.chat_input(
            placeholder="Who won the Women's U.S. Open in 2018?"
        ):
            st.chat_message(name="user").write(prompt)
            self._chain.predict(input=prompt)
            st.experimental_rerun()


if __name__ == "__main__":
    ChatBot().run()

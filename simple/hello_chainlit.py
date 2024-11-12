import chainlit as cl
from chainlit.types import ThreadDict
from icecream import ic
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain_openai import ChatOpenAI
from rich.pretty import pprint as pp


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Morning routine ideation",
            message="Can you help me create a personalized morning routine that would help increase my productivity throughout the day? Start by asking me about my current habits and what activities energize me in the morning.",
        ),
        cl.Starter(
            label="Explain superconductors",
            message="Explain superconductors like I'm five years old.",
        ),
        cl.Starter(
            label="Python script for daily email reports",
            message="Write a script to automate sending daily email reports in Python, and walk me through how I would set it up.",
        ),
        cl.Starter(
            label="Text inviting friend to wedding",
            message="Write a text asking a friend to be my plus-one at a wedding next month. I want to keep it super short and casual, and offer an out.",
        ),
    ]


@cl.on_chat_start
async def on_chat_start():
    model = ChatOpenAI(streaming=True)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You're a very knowledgeable historian who provides accurate and eloquent answers to historical questions.",
            ),
            ("human", "{question}"),
        ]
    )
    chain = prompt | model | StrOutputParser()
    cl.user_session.set("chain", chain)


@cl.on_message
async def on_message(message: cl.Message):
    pp(cl.chat_context.to_openai())
    ic(message.content)

    msg = cl.Message(content="")
    chain = cl.user_session.get("chain")

    for chunk in chain.stream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send() 


@cl.on_stop
async def on_stop():
    ic("on stop!")


@cl.on_chat_end
async def on_chat_end():
    print("session on end!")


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    print("session resume!")

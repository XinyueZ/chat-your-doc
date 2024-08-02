#
# Implemention of [Translation Agent: Agentic translation using reflection workflow](https://github.com/andrewyng/translation-agent/tree/main)
# with Llama-Index [workflow framework](https://www.llamaindex.ai/blog/introducing-workflows-beta-a-new-way-to-create-complex-ai-applications-with-llamaindex)
#

import asyncio
from typing import Optional

import streamlit as st
from icecream import ic
from langchain.prompts import (HumanMessagePromptTemplate,
                               SystemMessagePromptTemplate)
from langchain.schema.output_parser import StrOutputParser
from langchain_anthropic import ChatAnthropic
from langchain_community.llms.ollama import Ollama
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from llama_index.core.workflow import (Event, StartEvent, StopEvent, Workflow,
                                       draw_all_possible_flows,
                                       draw_most_recent_execution, step)
from rich.pretty import pprint as pp

MAX_TOKEN = 1024

LOOK_UP_LLM = {
    "Ollama/phi3": {
        "model": Ollama(
            model="phi3:latest",
            temperature=st.session_state.get(
                "key_slider_init_translation_temperature", 0.0
            ),
        ),
    },
    "Ollama/llama3.1": {
        "model": Ollama(
            model="llama3.1:latest",
            temperature=st.session_state.get(
                "key_slider_init_translation_temperature", 0.0
            ),
        ),
    },
    "Ollama/mistral": {
        "model": Ollama(
            model="mistral:latest",
            temperature=st.session_state.get(
                "key_slider_init_translation_temperature", 0.0
            ),
        ),
    },
    "claude-3-5-sonnet-20240620": {
        "model": ChatAnthropic(
            model="claude-3-5-sonnet-20240620",
            temperature=st.session_state.get(
                "key_slider_init_translation_temperature", 0.0
            ),
            max_tokens=MAX_TOKEN,
        ),
    },
    "gpt-4o": {
        "model": ChatOpenAI(
            model="gpt-4o",
            temperature=st.session_state.get("key_slider_reflection_temperature", 0.0),
            max_tokens=MAX_TOKEN,
        ),
    },
    "gpt-4o-mini": {
        "model": ChatOpenAI(
            model="gpt-4o-mini",
            temperature=st.session_state.get("key_slider_reflection_temperature", 0.0),
            max_tokens=MAX_TOKEN,
        ),
    },
    "gemini-1.5-flash-latest": {
        "model": ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            temperature=st.session_state.get(
                "key_slider_improve_translation_temperature", 0.0
            ),
            max_tokens=MAX_TOKEN,
        ),
    },
}


class InitTranslationEvent(Event):
    source_lang: str
    target_lang: str
    source_text: str
    country: str
    translation_1: str


class ReflectionEvent(Event):
    source_lang: str
    target_lang: str
    source_text: str
    country: str
    translation_1: str
    reflection: str


class ImproveTranslationEvent(Event):
    source_lang: str
    target_lang: str
    source_text: str
    country: str
    translation_1: str
    reflection: str
    translation_2: str


class AgenticTranslationWorkflow(Workflow):

    @step()
    async def do_init_translation(self, event: StartEvent) -> InitTranslationEvent:
        ic(event)
        system_message_prompt = "You are an expert linguist, specializing in translation from {source_lang} to {target_lang}."
        system_message_tmpl = SystemMessagePromptTemplate.from_template(
            template=system_message_prompt
        )

        human_message_prompt = """This is an {source_lang} to {target_lang} translation, please provide the {target_lang} translation for this text. \
        Do not provide any explanations or text apart from the translation.
        {source_lang}: {source_text}

        {target_lang}:"""
        human_message_tmpl = HumanMessagePromptTemplate.from_template(
            template=human_message_prompt
        )

        prompt = ChatPromptTemplate.from_messages(
            [system_message_tmpl, human_message_tmpl]
        )

        model = LOOK_UP_LLM[
            st.session_state.get("key_selection_init_translation_model", 0)
        ]["model"]

        chain = prompt | model | StrOutputParser()
        translation_1 = chain.invoke(
            {
                "source_lang": event["source_lang"],
                "target_lang": event["target_lang"],
                "source_text": event["source_text"],
            },
            {"configurable": {"session_id": None}},
        )
        ic(translation_1)
        return InitTranslationEvent(
            source_lang=event["source_lang"],
            target_lang=event["target_lang"],
            source_text=event["source_text"],
            country=event["country"],
            translation_1=translation_1,
        )

    @step()
    async def do_reflection(self, event: InitTranslationEvent) -> ReflectionEvent:
        ic(event)
        system_message_prompt = "You are an expert linguist specializing in translation from {source_lang} to {target_lang}. \
You will be provided with a source text and its translation and your goal is to improve the translation."
        system_message_tmpl = SystemMessagePromptTemplate.from_template(
            template=system_message_prompt
        )

        human_message_prompt = """Your task is to carefully read a source text and a translation from {source_lang} to {target_lang}, and then give constructive criticism and helpful suggestions to improve the translation. \
The final style and tone of the translation should match the style of {target_lang} colloquially spoken in {country}.

The source text and initial translation, delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT> and <TRANSLATION></TRANSLATION>, are as follows:

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation_1}
</TRANSLATION>

When writing suggestions, pay attention to whether there are ways to improve the translation's \n\
(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),\n\
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),\n\
(iii) style (by ensuring the translations reflect the style of the source text and take into account any cultural context),\n\
(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {target_lang}).\n\
Write a list of specific, helpful and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation.
Output only the suggestions and nothing else."""
        human_message_tmpl = HumanMessagePromptTemplate.from_template(
            template=human_message_prompt
        )

        prompt = ChatPromptTemplate.from_messages(
            [system_message_tmpl, human_message_tmpl]
        )

        model = LOOK_UP_LLM[st.session_state.get("key_selection_reflection_model", 1)][
            "model"
        ]

        chain = prompt | model | StrOutputParser()
        reflection = chain.invoke(
            {
                "source_lang": event.source_lang,
                "target_lang": event.target_lang,
                "source_text": event.source_text,
                "translation_1": event.translation_1,
                "country": event.country,
            },
            {"configurable": {"session_id": None}},
        )
        ic(reflection)
        return ReflectionEvent(
            source_lang=event.source_lang,
            target_lang=event.target_lang,
            source_text=event.source_text,
            country=event.country,
            translation_1=event.translation_1,
            reflection=reflection,
        )

    @step()
    async def do_improve_translation(self, event: ReflectionEvent) -> StopEvent:
        ic(event)
        system_message_prompt = "You are an expert linguist, specializing in translation editing from {source_lang} to {target_lang}."
        system_message_tmpl = SystemMessagePromptTemplate.from_template(
            template=system_message_prompt
        )

        human_message_prompt = """Your task is to carefully read, then edit, a translation from {source_lang} to {target_lang}, taking into
account a list of expert suggestions and constructive criticisms.

The source text, the initial translation, and the expert linguist suggestions are delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT>, <TRANSLATION></TRANSLATION> and <EXPERT_SUGGESTIONS></EXPERT_SUGGESTIONS> \
as follows:

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation_1}
</TRANSLATION>

<EXPERT_SUGGESTIONS>
{reflection}
</EXPERT_SUGGESTIONS>

Please take into account the expert suggestions when editing the translation. Edit the translation by ensuring:

(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules and ensuring there are no unnecessary repetitions), \
(iii) style (by ensuring the translations reflect the style of the source text)
(iv) terminology (inappropriate for context, inconsistent use), or
(v) other errors.

Output only the new translation and nothing else, no instructions, XML tags, headlines etc."""
        human_message_tmpl = HumanMessagePromptTemplate.from_template(
            template=human_message_prompt
        )

        prompt = ChatPromptTemplate.from_messages(
            [system_message_tmpl, human_message_tmpl]
        )

        model = LOOK_UP_LLM[
            st.session_state.get("key_selection_improve_translation_model", 0)
        ]["model"]

        chain = prompt | model | StrOutputParser()
        response = chain.invoke(
            {
                "source_lang": event.source_lang,
                "target_lang": event.target_lang,
                "source_text": event.source_text,
                "country": event.country,
                "translation_1": event.translation_1,
                "reflection": event.reflection,
            },
            {"configurable": {"session_id": None}},
        )
        translation_2 = (
            response if not isinstance(response, list) else response[0]["text"]
        )
        ic(translation_2)
        return StopEvent(
            result=ImproveTranslationEvent(
                source_lang=event.source_lang,
                target_lang=event.target_lang,
                source_text=event.source_text,
                country=event.country,
                translation_1=event.translation_1,
                reflection=event.reflection,
                translation_2=translation_2,
            )
        )


async def handle_translation():
    source_text = st.sidebar.text_area("Source text", key="key_textarea_source_text")
    if source_text is not None and len(source_text) > 10:
        if st.sidebar.button("translate"):
            workflow = AgenticTranslationWorkflow(timeout=60, verbose=True)
            event = {
                "source_lang": st.session_state.get("key_selection_source_language", 0),
                "target_lang": st.session_state.get("key_selection_target_language", 1),
                "source_text": source_text.strip(),
                "country": st.session_state.get("key_selection_country", 1),
            }
            pp(event)
            with st.spinner("Translating..."):
                result = await workflow.run(**event)
                with st.expander("Init translation"):
                    st.write(result.translation_1)
                with st.expander("Reflection"):
                    st.write(result.reflection)
                with st.expander("Improve translation"):
                    st.write(result.translation_2)


def model_options():
    st.write("Model options")
    st.selectbox(
        "Init transtation model",
        list(LOOK_UP_LLM.keys()),
        index=0,
        key="key_selection_init_translation_model",
    )
    st.slider("Temperature", 0.0, 1.0, key="key_slider_init_translation_temperature")

    st.selectbox(
        "Reflection model",
        list(LOOK_UP_LLM.keys()),
        index=1,
        key="key_selection_reflection_model",
    )
    st.slider("Temperature", 0.0, 1.0, key="key_slider_reflection_temperature")

    st.selectbox(
        "Improve translation model",
        list(LOOK_UP_LLM.keys()),
        index=0,
        key="key_selection_improve_translation_model",
    )
    st.slider(
        "Temperature",
        0.0,
        1.0,
        key="key_slider_improve_translation_temperature",
    )


def translation_options():
    st.write("Translation options")
    st.selectbox(
        "Source language",
        ["English", "Chinese", "German"],
        index=0,
        key="key_selection_source_language",
    )
    st.selectbox(
        "Target language",
        ["English", "Chinese", "German"],
        index=1,
        key="key_selection_target_language",
    )
    st.selectbox(
        "Country",
        ["United States", "China, Mainland", "Germany"],
        index=1,
        key="key_selection_country",
    )


async def main():
    st.sidebar.header("Translation Agent")
    st.write(
        """
    Implemention of [Translation Agent: Agentic translation using reflection workflow](https://github.com/andrewyng/translation-agent/tree/main)
    with Llama-Index [workflow framework](https://www.llamaindex.ai/blog/introducing-workflows-beta-a-new-way-to-create-complex-ai-applications-with-llamaindex)
    """
    )
    await handle_translation()
    st.sidebar.write("----")
    with st.sidebar:
        model_options()
        st.write("----")
        translation_options()


if __name__ == "__main__":
    asyncio.run(main())

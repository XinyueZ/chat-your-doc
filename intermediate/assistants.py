import os

import streamlit as st
from langchain.agents.openai_assistant.base import OpenAIAssistantRunnable, OutputType
from langchain.chat_models import ChatOpenAI
from loguru import logger
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

from openai import OpenAI
from openai.types import FileObject

st.set_page_config(layout="wide")


class App:
    def __call__(self):
        approach = st.sidebar.radio(
            "# Approach",
            [
                "LangChain",
                "SDK",
                "REST",
            ],
            index=0,
        )
        openai_api_key = os.environ["OPENAI_API_KEY"]
        model_name = "gpt-4-1106-preview"

        if approach == "LangChain":
            st.write("# Code generation assistant")
            inputs = st.text_input(
                "Coding challenge(Press Enter)",
                placeholder="Find odd numbers from 1 to 100",
            )
            if inputs != None and inputs != "":
                instructions = """As a software developer, your task is to devise solutions to the presented problems and provide implementable code. In cases where the programming language isn't explicitly mentioned, please default to using Python.

                                Notice: Only returns code and associated comments in certain programming languages.
                                """
                tools = [{"type": "code_interpreter"}]
                outputs, structured_output_parser = self._do_in_langchain(
                    inputs,
                    instructions,
                    tools,
                    model_name,
                )
                if outputs != None:
                    self._langchain_outputs(outputs, structured_output_parser)
        elif approach == "SDK":
            openai = OpenAI(api_key=openai_api_key)
            file_path = self._upload_doc()
            logger.debug(f"file_path: {file_path}")
            if file_path != None:
                uploaded_file = self._upload2openai(openai, file_path)
                logger.debug(f"Uploaded file: {uploaded_file}")
            st.write("# Knowledge assistant")
            inputs = st.text_input("Query(Press Enter)", placeholder="What is NeRF?")
            if inputs != None and inputs != "":
                instructions = """As a knowledge provider, your task is to provide the information requested based on the files you have."""
                tools = [{"type": "retrieval"}]
                outputs = self._do_with_sdk(
                    openai,
                    inputs,
                    instructions,
                    tools,
                    model_name,
                )
                if outputs != None:
                    st.write(outputs)

    def _langchain_outputs(
        self,
        outputs: OutputType,
        structured_output_parser: StructuredOutputParser,
    ):
        struct_output = structured_output_parser.parse(outputs[0].content[0].text.value)
        block = struct_output["code"]["block"]
        lang = struct_output["code"]["lang"]
        explain = struct_output["code"]["explain"]

        st.write(f"> Code in {lang}")
        st.code(block, language=lang.lower(), line_numbers=True)
        st.write(explain)

    def _do_in_langchain(
        self,
        inputs: str,
        instructions: str,
        tools: list[dict[str, str]],
        model_name: str,
    ) -> tuple[OutputType, StructuredOutputParser]:
        structured_output_parser = StructuredOutputParser.from_response_schemas(
            [
                ResponseSchema(
                    name="code",
                    description="""A code snippet block, programming language name and explaination:
                            { "block": string // the code block,  "lang": string // programming langauge name, "explain": string // explaination of the code}
                            """,
                    type="string",
                ),
            ]
        )
        assis = OpenAIAssistantRunnable.create_assistant(
            "Coding Assistant",
            instructions=instructions
            + "\n\n"
            + f"The format instructions is {structured_output_parser.get_format_instructions()}",
            tools=tools,
            model=model_name,
        )
        outputs = assis.invoke({"content": inputs})
        logger.debug(outputs)
        return outputs, structured_output_parser

    def _upload_doc(self) -> str | None:
        with st.sidebar:
            uploaded_doc = st.file_uploader("Upload document", key="doc_uploader")
            if not uploaded_doc:
                return None
            if uploaded_doc:
                tmp_dir = "tmp/"
                if not os.path.exists(tmp_dir):
                    os.makedirs(tmp_dir)
                temp_file_path = os.path.join(tmp_dir, f"{uploaded_doc.name}")
                with open(temp_file_path, "wb") as file:
                    file.write(uploaded_doc.getvalue())
                    file_name = uploaded_doc.name
                    logger.debug(f"Uploaded {file_name}")
                    uploaded_doc.flush()
                    uploaded_doc.close()
                    return temp_file_path
                    # os.remove(temp_file_path)

    def _do_with_sdk(
        self,
        openai: OpenAI,
        inputs: str,
        instructions: str,
        tools: list[dict[str, str]],
        model_name: str,
    ) -> str | None:
        uploaded_files = openai.files.list()
        uploaded_files_data = uploaded_files.data
        uploaded_fileids = list(map(lambda x: x.id, uploaded_files_data))
        logger.debug(uploaded_fileids)

        assis = openai.beta.assistants.create(
            name="Knowledge Assistant",
            instructions=instructions,
            model=model_name,
            tools=tools,
            file_ids=uploaded_fileids,
        )

        thread = openai.beta.threads.create()

        openai.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=inputs,
        )

        run = openai.beta.threads.runs.create(
            thread_id=thread.id, assistant_id=assis.id
        )

        while True:
            retrieved_run = openai.beta.threads.runs.retrieve(
                thread_id=thread.id, run_id=run.id
            )
            logger.debug(retrieved_run)
            if retrieved_run.status == "completed":
                break

        thread_messages = openai.beta.threads.messages.list(thread.id)
        logger.debug(thread_messages.data)

        return thread_messages.data[0].content[0].text.value

    def _upload2openai(self, openai: OpenAI, file_path: str) -> FileObject:
        filename = os.path.basename(file_path)
        file_objects = list(
            filter(lambda x: x.filename == filename, openai.files.list().data)
        )
        if len(file_objects) > 0:
            return None
        uploaded_file = openai.files.create(
            file=open(file_path, "rb"),
            purpose="assistants",
        )
        return uploaded_file


if __name__ == "__main__":
    app = App()
    app()

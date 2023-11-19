import os, sys
import streamlit as st
from langchain.agents.openai_assistant.base import OpenAIAssistantRunnable, OutputType
from langchain.chat_models import ChatOpenAI
from loguru import logger
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

st.set_page_config(layout="wide")


class App:
    def __call__(self):
        st.write("# Code generation assistant")
        # A radio button group for selecting the approaches like: langchain, sdk or REST
        approach = st.sidebar.radio(
            "# Approach",
            [
                "LangChain",
                "SDK",
                "REST",
            ],
            index=0,
        )

        model_name = "gpt-4-1106-preview"

        if approach == "LangChain":
            inputs = st.text_input(
                "Prompt(Press Enter)", value="Find odd numbers from 1 to 100"
            )
            instructions = """As a software developer, your task is to devise solutions to the presented problems and provide implementable code. In cases where the programming language isn't explicitly mentioned, please default to using Python.

            Notice: Only returns code and associated comments in certain programming languages.
            """
            tools = [{"type": "code_interpreter"}]
            if inputs != None and inputs != "":
                outputs, structured_output_parser = self._do_in_langchain(
                    inputs,
                    instructions,
                    tools,
                    model_name,
                )
                if outputs != None:
                    self._langchain_outputs(outputs, structured_output_parser)

    def _langchain_outputs(self, outputs, structured_output_parser):
        struct_output = structured_output_parser.parse(
            outputs[-1].content[-1].text.value
        )
        block = struct_output["code"]["block"]
        lang = struct_output["code"]["lang"]
        explain = struct_output["code"]["explain"]
        st.write(f"> Code in {lang}")
        st.code(block, language=lang, line_numbers=True)
        st.write(explain)

    def _do_in_langchain(
        self, inputs, instructions, tools, model_name
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
            "Assistant Bot",
            instructions=instructions
            + "\n\n"
            + f"The format instructions is {structured_output_parser.get_format_instructions()}",
            tools=tools,
            model=model_name,
        )
        outputs = assis.invoke({"content": inputs})
        logger.debug(outputs)
        return outputs, structured_output_parser

    def _upload_doc(self) -> None:
        with st.sidebar:
            uploaded_doc = st.file_uploader("Upload document")
            if uploaded_doc:
                tmp_dir = "tmp/"
                if not os.path.exists(tmp_dir):
                    os.makedirs(tmp_dir)
                temp_file_path = os.path.join(tmp_dir, f"{uploaded_doc.name}")
                with open(temp_file_path, "wb") as file:
                    file.write(uploaded_doc.getvalue())
                    file_name = uploaded_doc.name
                    logger.debug(f"Uploaded {file_name}")

                # os.remove(temp_file_path)


if __name__ == "__main__":
    app = App()
    app()

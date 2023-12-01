import http.client
import json
import os
import sys

import streamlit as st
from langchain.agents.openai_assistant.base import OpenAIAssistantRunnable, OutputType
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from loguru import logger
from openai import OpenAI
from openai.types import FileObject

logger.remove()
logger.add(sys.stderr, level="INFO")

st.set_page_config(layout="wide")


class DeTranslator:
    def __init__(self) -> None:
        self.sys_msg_tmpl = SystemMessagePromptTemplate.from_template(
            "You're a machine language translator. You can translate text of any language into German."
        )
        self.human_msg_tmpl = HumanMessagePromptTemplate.from_template(
            "Translate this sentence: {sentence}"
        )

        self.chat_prompt = ChatPromptTemplate.from_messages(
            [self.sys_msg_tmpl, self.human_msg_tmpl]
        )

        self.llm_chain = LLMChain(llm=ChatOpenAI(), prompt=self.chat_prompt)

    def __call__(self, text: str = None):
        res = self.llm_chain.run(
            sentence=text,
        )
        logger.debug("Translated: " + res)
        return res


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
        elif approach == "REST":
            st.write("# Print assistant")
            inputs = st.text_input("Print(Press Enter)", placeholder="Hello,world")
            if inputs != None and inputs != "":
                instructions = """As an assistant to translate any text into German langauge and print out."""
                outputs: tuple[str, str] | None = self._do_in_rest(
                    openai_api_key,
                    inputs,
                    instructions,
                    model_name,
                )
                if outputs != None:
                    result, run_res = outputs[0], outputs[1]
                    st.write(result)
                    st.write("---")
                    st.write(run_res)
        else:
            pass

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

    def _get(self, key: str, host: str, url: str):
        conn = http.client.HTTPSConnection(host)
        headers = {
            "Content-Type": "application/json",
            "OpenAI-Beta": "assistants=v1",
        }
        auth_header = f"Bearer {key}"
        headers["Authorization"] = auth_header
        conn.request("GET", url, None, headers)
        res = conn.getresponse()

        logger.debug(f"Status: {res.status}")
        logger.debug(type(res))
        logger.debug(f"Header: {res.headers}")
        logger.debug(f"Response: {res}")
        res_content = res.read().decode("utf-8").strip("\ufeff")
        logger.debug(f"Response: {res_content}")

        reply = json.loads(res_content, strict=False)
        conn.close()
        return reply

    def _post(self, key: str, host: str, url: str, data: str):
        # cmd = [
        #     "curl",
        #     api,
        #     f"-u:{key}",
        #     '-H "Content-Type: application/json"',
        #     '-H "OpenAI-Beta: assistants=v1"',
        #     f"-d '{data}'",
        # ]

        # logger.debug(cmd)
        # result = subprocess.check_output(" ".join(cmd))
        # result = os.system(" ".join(cmd))
        # result = run(" ".join(cmd), stdout=PIPE, stderr=PIPE, universal_newlines=True)
        # logger.debug(result)

        conn = http.client.HTTPSConnection(host)
        headers = {
            "Content-Type": "application/json",
            "OpenAI-Beta": "assistants=v1",
        }
        payload = data
        auth_header = f"Bearer {key}"
        headers["Authorization"] = auth_header
        conn.request("POST", url, payload, headers)
        res = conn.getresponse()

        logger.debug(f"Status: {res.status}")
        logger.debug(type(res))
        logger.debug(f"Header: {res.headers}")
        logger.debug(f"Response: {res}")
        res_content = res.read().decode("utf-8").strip("\ufeff")
        logger.debug(f"Response: {res_content}")

        reply = json.loads(res_content, strict=False)
        conn.close()
        return reply

    def _do_in_rest(
        self,
        key: str,
        inputs: str,
        instructions: str,
        model_name: str,
    ) -> tuple[str, str] | None:
        translator_def = """"function": {
            "name": "translate_to_de_string",
            "description": "Use this function to translate a string into German",
            "parameters": {
                "type": "object",
                "properties": {
                    "string": {
                        "type": "string",
                        "description": "Any string input by user to translate"
                    }
                },
                "required": ["string"]
            }
        }"""
        print_func_def = """"function": {
            "name": "print_string",
            "description": "Use this function to print any string performed by translate_to_de_string",
            "parameters": {
                "type": "object",
                "properties": {
                    "string": {
                        "type": "string",
                        "description": "Any string input by user to print"
                    }
                },
                "required": ["string"]
            }
        }"""

        assis_create = (
            "{\n"
            '    "name": "Print Assistant",\n'
            '    "model": "' + model_name + '",\n'
            '    "instructions": "' + instructions + '",\n'
            '    "tools": [\n'
            "        {\n"
            '            "type": "function",\n'
            f"            {translator_def} \n"
            "        },\n"
            "        {\n"
            '            "type": "function",\n'
            f"            {print_func_def} \n"
            "        }\n"
            "    ]\n"
            "}"
        )
        assis_res = self._post(
            key,
            "api.openai.com",
            "/v1/assistants",
            assis_create,
        )
        logger.debug(assis_res)
        assis_id = assis_res["id"]

        thread_create = ""
        thread_res = self._post(
            key,
            "api.openai.com",
            "/v1/threads",
            thread_create,
        )
        logger.debug(thread_res)
        thread_id = thread_res["id"]

        content = '"' + inputs.strip() + '"'
        message_create = '{"role": "user", "content":' + content + "}"
        message_res = self._post(
            key,
            "api.openai.com",
            f"/v1/threads/{thread_id}/messages",
            message_create,
        )
        logger.debug(message_res)
        # Warning, message must be created before a "run" is activated.

        run_create = '{"assistant_id": "' + assis_id + '"}'
        run_res = self._post(
            key,
            "api.openai.com",
            f"/v1/threads/{thread_id}/runs",
            run_create,
        )
        logger.debug(run_res)
        run_id = run_res["id"]
        run_status = run_res["status"]

        try:
            result = None
            while True:
                logger.debug("🏃‍♂️ Run status: " + run_status)
                run_res = self._get(
                    key,
                    "api.openai.com",
                    f"/v1/threads/{thread_id}/runs/{run_id}",
                )
                run_status = run_res["status"]
                if run_status == "completed":
                    break
                elif run_status == "requires_action":
                    logger.debug("🏃‍♂️ Run status: " + run_status)

                    required_action = run_res["required_action"]
                    required_action_type = required_action["type"]
                    if required_action_type != "submit_tool_outputs":
                        continue
                    submit_tool_outputs = required_action["submit_tool_outputs"]
                    tool_calls = submit_tool_outputs["tool_calls"]
                    tool_call_func = tool_calls[0]["function"]
                    func_name = tool_call_func["name"]
                    arguments = tool_call_func["arguments"]
                    result = json.loads(arguments)["string"]

                    tool_call_id = tool_calls[0]["id"]
                    logger.debug(f"Tool call id: {tool_call_id}")

                    if func_name == "translate_to_de_string":
                        de_translator = DeTranslator()
                        result = de_translator(result)
                    elif func_name == "print_string":
                        # We don't need to really print, just assign the 
                        # result and return to outside.
                        result = f"{result}"

                    submit_tool_output = (
                        '{"tool_call_id":'
                        + '"'
                        + tool_call_id
                        + '",'
                        + '"output":'
                        + '"'
                        + result
                        + '"}'
                    )

                    submit_tool_outputs: str = (
                        '{"tool_outputs":[' + submit_tool_output + "]}"
                    )
                    # Important, otherwise there is server-error 400 while POST.
                    submit_tool_outputs = submit_tool_outputs.encode()

                    logger.debug(submit_tool_outputs)

                    submit_res = self._post(
                        key,
                        "api.openai.com",
                        f"/v1/threads/{thread_id}/runs/{run_id}/submit_tool_outputs",
                        submit_tool_outputs,
                    )
                    logger.debug(submit_res)
            return result, run_res
        except Exception as e:
            logger.error(e)
            return None


if __name__ == "__main__":
    App()()

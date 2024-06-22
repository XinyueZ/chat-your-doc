import asyncio
import base64
import io
import os
import sys
import uuid
from datetime import datetime
from functools import partial
from inspect import getframeinfo, stack
from typing import Any, Dict, List, Tuple

import filetype
import nest_asyncio
import numpy as np
import pandas as pd
import requests
import streamlit as st
import torch
import yfinance
from audiocraft.data.audio import audio_write
from audiocraft.models import musicgen
from audiocraft.utils.notebook import display_audio
from langchain import hub
from langchain.agents import Tool, load_tools
from langchain.agents.agent import AgentExecutor
from langchain.agents.structured_chat.base import create_structured_chat_agent
from langchain.memory import ChatMessageHistory
from langchain.prompts import HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.wikidata.tool import (WikidataAPIWrapper,
                                                     WikidataQueryRun)
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.vectorstores import FAISS
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.tool import ToolMessage, ToolMessageChunk
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from matplotlib import pyplot as plt
from mutagen.mp3 import MP3
from openai import OpenAI
from PIL import Image, UnidentifiedImageError
from rich.pretty import pprint
from transformers import pipeline
from ultralytics import YOLO
from ultralytics.engine.results import Boxes, Results

VERBOSE = True
MAX_TOKEN = 2048

OPENAI_LLM = "gpt-4o"
GOOGLE_LLM = "gemini-1.5-flash-latest"
CV_MODEL: str = "yolov8s-world.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FUN_MAPPING = {}

nest_asyncio.apply()

st.set_page_config(layout="wide")
# ------------------------------------------helpers------------------------------------------


def pretty_print(title: str = "Untitled", content: Any = None):
    if not VERBOSE:
        return

    info = getframeinfo(stack()[1][0])
    print()
    pprint(
        f":--> {title} --> {info.filename} --> {info.function} --> line: {info.lineno} --:"
    )
    pprint(content)


def image_url_to_image(image_url: str) -> Image:
    return Image.open(requests.get(image_url, stream=True).raw)


def create_random_filename(ext=".txt") -> str:
    return create_random_name() + ext


def create_random_name() -> str:
    return str(uuid.uuid4())


# enum class for audio or image
class MediaType:
    AUDIO = "audio"
    IMAGE = "image"


def doc_uploader() -> Tuple[bytes, str]:
    with st.sidebar:
        uploaded_doc = st.file_uploader(
            "# Upload one image or an audio file", key="doc_uploader"
        )
        if not uploaded_doc:
            st.session_state["file_name"] = None
            st.session_state["base64_object"] = None
            # pretty_print("doc_uploader", "No image uploaded")
            return None
        if uploaded_doc:
            tmp_dir = "./tmp"
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir)
            temp_file_path = os.path.join(tmp_dir, f"{uploaded_doc.name}")
            with open(temp_file_path, "wb") as file:
                file.write(uploaded_doc.getvalue())
                file_name = uploaded_doc.name
                media_type = (
                    MediaType.IMAGE
                    if filetype.is_image(temp_file_path)
                    else MediaType.AUDIO
                )

                # pretty_print("doc_uploader", f"Uploaded {file_name}")
                uploaded_doc.flush()
                uploaded_doc.close()
                # os.remove(temp_file_path)
                if st.session_state.get("file_name") == file_name:
                    # pretty_print("doc_uploader", "Same file")
                    return st.session_state["base64_object"]

                # pretty_print("doc_uploader", "New file")
                st.session_state["file_name"] = temp_file_path
                with open(temp_file_path, "rb") as image_file:
                    st.session_state["base64_object"] = base64.b64encode(
                        image_file.read()
                    ).decode("utf-8")

                return st.session_state["base64_object"], media_type
        return None


# ------------------------------------------LLM Chain------------------------------------------


def create_message_chain(
    model: BaseChatModel,
    base64_object: bytes,
    media_type: str,
) -> Runnable:
    handle_image = media_type == MediaType.IMAGE
    handle_audio = media_type == MediaType.AUDIO
    pretty_print("handle_image", handle_image)
    pretty_print("handle_audio", handle_audio)

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=[
                    {
                        "type": "text",
                        "text": """As a helpful assistant, you should respond to the user's query.
Avoid giving any information related to the local file system or sandbox.""",
                    }
                ]
            ),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template(
                template=(
                    [
                        {
                            "type": "text",
                            "text": "{query}",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_object}",
                            },
                        },
                    ]
                    if handle_image
                    else [
                        {"type": "text", "text": "{query}"},
                    ]
                )
            ),
        ]
    )
    return prompt | model


def create_tool_chain(model: BaseChatModel) -> Runnable:
    return model


# ------------------------------------------agent, functions, tools------------------------------------------
class LoadUrlsTool(BaseModel):
    query: str = Field(description="The query to search for.")
    urls: List[str] = Field(description="The URLs to load.")


@tool("load-urls-tool", args_schema=LoadUrlsTool, return_direct=False)
def load_urls_tool(retrieval_model: BaseChatModel, query: str, urls: List[str]) -> str:
    """Load the content of the given Urls for getting responses to the query and return the query result."""

    load_urls_prompt_template = PromptTemplate.from_template(
        """Reponse the query inside [query] based on the context inside [context]:
[query]
{query}
[query]

[context]
{context}
[context]

Only return the answer without any instruction text or additional information.
Keep the result as simple as possible."""
    )
    loader = WebBaseLoader(urls)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0
    )
    chunks = splitter.split_documents(docs)
    db = FAISS.from_documents(chunks, OpenAIEmbeddings())
    retriever = db.as_retriever()

    chain = (
        {"context": retriever, "query": RunnablePassthrough()}
        | load_urls_prompt_template
        | retrieval_model
        | StrOutputParser()
    )
    return chain.invoke(query)


search_agent_tools = [
    Tool(
        name="Google Search",
        description="Search Google for recent results.",
        func=GoogleSearchAPIWrapper(
            google_api_key=os.environ.get("GOOGLE_CSE_KEY")
        ).run,
    ),
    Tool(
        name="DuckDuckGo Search",
        func=DuckDuckGoSearchRun().run,
        description="Use to search for the information from DuckDuckGo.",
    ),
    Tool(
        name="Wikipedia Search",
        func=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()).run,
        description="Use to search for the information from Wikipedia.",
    ),
    Tool(
        name="Wikidata Search",
        func=WikidataQueryRun(api_wrapper=WikidataAPIWrapper()).run,
        description="Use to search for the information from Wikidata.",
    ),
]
search_agent_tools.extend(load_tools(["arxiv"]))


def create_search_agent(agent_model: BaseChatModel) -> AgentExecutor:
    return AgentExecutor(
        agent=create_structured_chat_agent(
            llm=agent_model,
            tools=search_agent_tools,
            prompt=hub.pull("hwchase17/structured-chat-agent"),
        ),
        tools=search_agent_tools,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )


# ------------------------------------------tool functions------------------------------------------


def generate_image(model: BaseChatModel, context: str) -> str:
    """Generate an image to illustrate for user request."""

    tools = load_tools(["dalle-image-generator"])
    agent = create_structured_chat_agent(
        llm=model,
        tools=tools,
        prompt=hub.pull("hwchase17/structured-chat-agent"),
    )
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )
    prompt = f"""Generate an image to illustrate for user request with the following context:

Context:
{context}

Notice: 
- ONLY return the image link.
- WHEN the image includes text, it MUST be in the same language as the language of the input text.
"""
    image_gen = agent_executor.invoke({"input": prompt})
    # pretty_print("Image Gen:", image_gen)
    try:
        image_url = image_gen["output"]
        return image_url_to_image(image_url), image_url
    except UnidentifiedImageError:
        return None, None


def annotate_image(
    model: BaseChatModel,
    base64_image: bytes,
    image_description: str,
) -> Tuple[Results, str]:
    """Annotate an image with classes (COCO dataset types), response results with bounding boxes."""

    def coco_label_extractor(model: BaseChatModel, image_description: str) -> str:
        """Read an image description and extract COCO defined labels as much as possible from the description."""
        template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You as an AI assistant can understand an image descritpion. 
                 Try to extract COCO defined labels as much as possible from the description.
                 Only return lables and split by comma, no empty space.""",
                ),
                ("human", "Image descritpion: {img_desc}"),
            ]
        )
        human_input = template.format_messages(img_desc=image_description)
        return model.invoke(human_input).content

    # pretty_print("Image description:", image_description)
    classes = coco_label_extractor(model, image_description)
    # pretty_print("Classes:", classes)
    classes = classes.split(",") if classes else list()
    model = YOLO(CV_MODEL)  # or select yolov8m/l-world.pt for different sizes

    if classes is not None and len(classes) > 0:
        model.set_classes(classes)

    image = Image.open(io.BytesIO(base64.b64decode(base64_image)))
    preds = model.predict(image)
    results: Results = preds[0]
    save_dir = "tmp"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = f"{save_dir}/annotated_{create_random_filename('.jpg')}"
    results.save(save_path)
    return results, save_path


def run_search_agent(agent: AgentExecutor, topic: str) -> Dict[str, Any]:
    """Run the agent for the topic, the agent will search the topic through the web and return the result."""
    prompt = f"""Search through the internet for the topic inside >>>>>>>>> and <<<<<<<<<<
>>>>>>>>>
{topic}
<<<<<<<<<<
We recommend certain tools that you can use:
-  Google Search
-  DuckDuckGo Search
-  Wikipedia Search
-  Wikidata Search
-  arxiv Search
Also if you face some urls for more details you can also use the tool:
- load-urls-tool
"""
    return agent.invoke({"input": prompt})


def get_current_time(_: str) -> str:
    """For questions or queries that are relevant to current time or date"""

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return current_time


def get_and_plot_stock_prices(
    stock_symbols: List[str], start_date: str, end_date: str
) -> Tuple[pd.DataFrame, str]:
    """Get and plot the stock prices for the given stock symbols between
    the start and end dates.

    Args:
        stock_symbols (str or list): The stock symbols to get the
        prices for.
        start_date (str): The start date in the format
        'YYYY-MM-DD'.
        end_date (str): The end date in the format 'YYYY-MM-DD'.

    Returns:
            Tuple:
                pandas.DataFrame: The stock prices for the given stock symbols indexed by date, with one column per stock symbol.
                stock_prices (pandas.DataFrame): The stock prices for the
                given stock symbols.
    """

    def _plot_stock_prices(stock_prices: pd.DataFrame, filename: str):
        """Plot the stock prices for the given stock symbols.
        Args:
            stock_prices (pandas.DataFrame): The stock prices for the given stock symbols.
            filename (str): The filename to save the plot to.

        """
        plt.figure(figsize=(10, 5))
        for column in stock_prices.columns:
            plt.plot(stock_prices.index, stock_prices[column], label=column)
        plt.title("Stock Prices")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend(stock_prices)
        plt.grid(True)
        plt.savefig(filename)

    stock_data = yfinance.download(stock_symbols, start=start_date, end=end_date)
    pretty_print("stock_data:", stock_data)
    prices_df = stock_data.get("Close")
    pretty_print("Close:", prices_df)

    plot_filename = create_random_filename(ext=".png")
    fullpath = f"./tmp/{plot_filename}"

    _plot_stock_prices(prices_df, fullpath)
    return (prices_df, fullpath)


def text2speech(text: str, voice_type: str = "alloy") -> str:
    """Convert text to speech, return the full filepath to the speech."""
    client = OpenAI()
    random_filename = create_random_filename(".mp3")
    speech_file_path = f"./tmp/{random_filename}"
    response = client.audio.speech.create(
        model="tts-1",
        voice=voice_type,
        input=text,
    )
    response.stream_to_file(speech_file_path)
    return speech_file_path


def text2music(prompt: str, duration=15) -> str:
    """Convert prompting text to music, return the full filepath to the music."""

    def _write_wav(output, file_initials):
        try:
            for idx, one_wav in enumerate(output):
                audio_write(
                    f"{file_initials}_{idx}",
                    one_wav.cpu(),
                    model.sample_rate,
                    strategy="loudness",
                    loudness_compressor=True,
                )
            return True
        except Exception as e:
            print("Error while writing the file ", e)
            return None

    model = musicgen.MusicGen.get_pretrained("medium", device=DEVICE)
    model.set_generation_params(duration=15)
    musicgen_out = model.generate([prompt], progress=True)
    musicgen_out_filename = f"./tmp/{create_random_name()}"
    _write_wav(musicgen_out, musicgen_out_filename)
    return f"{musicgen_out_filename}_0.wav"


def speech2text(base64_speech_audio: bytes) -> str:
    """Convert base64_speech_audio to text, extract the base64_speech_audio content, and provide the text as return."""
    audio_file_name = "./tmp/temp.wav"
    audio_file = open(audio_file_name, "wb")
    bytes_io = io.BytesIO(base64.b64decode(base64_speech_audio))
    audio_file.write(bytes_io.read())
    torch_dtype = torch.float32 if DEVICE == "cpu" else torch.float16
    pipe_aud2txt = pipeline(
        "automatic-speech-recognition",
        "openai/whisper-large-v3",
        torch_dtype=torch_dtype,
        device=DEVICE,
    )
    res = pipe_aud2txt(audio_file_name)
    return res.get("text")


def synthesize_audio(
    speech_text: str, prompt: str, voice_type: str = "alloy", duration=15
) -> str:
    """Generate an audio using the provided speech_text and background music audio generated from the prompt.
    Generate an audio from the speech_text and another background music from the prompt, then combine the two audioes
    into a single synthesis.
    """
    text2speech_file_fullpath = text2speech(speech_text, voice_type)
    text2music_file_fullpath = text2music(prompt, duration)
    synthesis_filename = f"./tmp/{create_random_filename('.mp3')}"

    os.system(
        f"ffmpeg -i {text2speech_file_fullpath} -stream_loop -1 -i {text2music_file_fullpath}  -filter_complex amix=inputs=2:duration=first:dropout_transition=2 {synthesis_filename} -y"
    )
    return synthesis_filename


# ------------------------------------------tool------------------------------------------
class GenerateImageTool(BaseModel):
    """Generate an image to illustrate for user request."""

    context: str = Field(
        ...,
        description="The context for generating an image to illustrate what the user requested.",
    )


class AnnotateImageTool(BaseModel):
    """Annotate an image with classes (COCO dataset types), response results with bounding boxes."""

    image_description: str = Field(
        ...,
        description="The description of the image that needs to be annotated.",
    )


class RunSearchAgentTool(BaseModel):
    """Run the agent to search for corresponding topic. The agent will search the web for the topic and return the results."""

    topic: str = Field(
        ...,
        description="The topic to search through the web.",
    )


class GetCurrentTimeTool(BaseModel):
    """Get the current time."""

    event: str = Field(
        ...,
        description="Some questions or queries that are relevant to current time or date",
    )


class GetAndPlotStockPrices(BaseModel):
    """Get and plot the stock prices for the given stock symbols between the start and end dates."""

    stock_symbols: str = Field(
        ...,
        description="The stock symbols list string to get prices for, should be a string with comma-separated different symbols.",
    )
    start_date: str = Field(
        ...,
        description="The start date in the format 'YYYY-MM-DD' or other understandable format.",
    )
    end_date: str = Field(
        ...,
        description="The end date in the format 'YYYY-MM-DD' or other understandable format.",
    )


class Text2SpeechTool(BaseModel):
    """Convert text to speech."""

    text: str = Field(
        ...,
        description="The text that will be converted to speech.",
    )


class Text2MusicTool(BaseModel):
    """Tool for generating music via text prompting."""

    prompt: str = Field(
        ...,
        description="Use the prompt text to generate music.",
    )


class Speech2TextTool(BaseModel):
    """Extract speech audio content to text."""

    pass


class SynthesizeAudioTool(BaseModel):
    """Synthesize audio based on the provided speech_text and prompt."""

    speech_text: str = Field(
        ...,
        description="The text that will be converted to speech.",
    )
    prompt: str = Field(
        None,
        description="The prompt that will be used to generate background music audio, if not set, the prompt will be the same as the speech_text.",
    )


# ------------------------------------------model event handlers------------------------------------------
def handle_generate_image(
    tool_name: str,
    tool_id: str,
    context: str,
) -> ToolMessage:
    additional_kwargs = dict()
    try:
        func = FUN_MAPPING.get(tool_name, None)
        if func:
            image, image_url = func(context=context)
        else:
            raise ValueError("No tool provided.")

        additional_kwargs = {"image_url": image_url} if image_url else {}
        return ToolMessage(
            content=(
                f"Generated image successfully, tool has finished: ![]({image_url})"
                if image_url
                else "Tool called, nothing was generated"
            ),
            tool_call_id=tool_id,
            additional_kwargs=additional_kwargs,
        )

    except Exception as e:
        st.write(f"Something went wrong.\n\n{e}")
        return ToolMessage(
            content=f"Tool was called but failed to generate image.\n\n{e}",
            tool_call_id=tool_id,
            additional_kwargs=additional_kwargs,
        )


def handle_annotate_image(
    tool_name: str,
    tool_id: str,
    base64_image: bytes,
    image_description: str,
) -> ToolMessage:
    additional_kwargs = {}
    try:
        func = FUN_MAPPING.get(tool_name, None)
        if func:
            _, image_path = func(
                base64_image=base64_image, image_description=image_description
            )
        else:
            raise ValueError("No tool provided.")

        additional_kwargs["image_path"] = image_path if image_path else ""
        return ToolMessage(
            content=(
                f"Annotated image successfully, tool has finished."
                if image_path
                else "Tool called, nothing was annotated"
            ),
            tool_call_id=tool_id,
            additional_kwargs=additional_kwargs,
        )

    except Exception as e:
        st.write(f"Something went wrong.\n\n{e}")
        return ToolMessage(
            content=f"Tool was called but failed to annotate image.\n\n{e}",
            tool_call_id=tool_id,
            additional_kwargs=additional_kwargs,
        )


def handle_search_agent(
    tool_name: str,
    tool_id: str,
    topic: str,
) -> ToolMessage:
    additional_kwargs = {}
    try:
        func = FUN_MAPPING.get(tool_name, None)
        if func:
            agent_res = func(topic)
        else:
            raise ValueError("No tool provided.")

        additional_kwargs["string"] = agent_res["output"]
        return ToolMessage(
            content=(
                f"Searched successfully, tool has finished:\n\n{agent_res['output']}"
                if agent_res["output"] and agent_res["output"] != ""
                else "Tool called, nothing was responsed by agent."
            ),
            tool_call_id=tool_id,
            additional_kwargs=additional_kwargs,
        )

    except Exception as e:
        st.write(f"Something went wrong.\n\n{e}")
        return ToolMessage(
            content=f"Tool was called but failed to get result.\n\n{e}",
            tool_call_id=tool_id,
            additional_kwargs=additional_kwargs,
        )


def handle_get_current_time(
    tool_name: str,
    tool_id: str,
) -> ToolMessage:
    additional_kwargs = {}
    try:
        func = FUN_MAPPING.get(tool_name, None)
        if func:
            current_time = func("")
        else:
            raise ValueError("No tool provided.")

        additional_kwargs["string"] = f"Current time: {current_time}"
        return ToolMessage(
            content=(
                f"Get current time: {current_time}, tool has finished."
                if current_time and current_time != ""
                else "Tool called, nothing was responsed by agent."
            ),
            tool_call_id=tool_id,
            additional_kwargs=additional_kwargs,
        )

    except Exception as e:
        st.write(f"Something went wrong.\n\n{e}")
        return ToolMessage(
            content=f"Tool was called but failed to get current time.\n\n{e}",
            tool_call_id=tool_id,
            additional_kwargs=additional_kwargs,
        )


def handle_get_stock_prices(
    tool_name: str,
    tool_id: str,
    tool_stock_symbols: str,
    tool_start_date: str,
    tool_end_date: str,
) -> ToolMessage:
    additional_kwargs = {}
    try:
        func = FUN_MAPPING.get(tool_name, None)
        if func:
            prices_df, image_path = func(
                stock_symbols=tool_stock_symbols.split(","),
                start_date=tool_start_date,
                end_date=tool_end_date,
            )
        else:
            raise ValueError("No tool provided.")

        additional_kwargs["image_path"], additional_kwargs["dataframe"] = (
            image_path,
            prices_df,
        )

        return ToolMessage(
            content=f"""Complete getting stock prices successfully, tool has finished, stop further investigation. The stock prices requested:
dataframe:
{prices_df}""",
            tool_call_id=tool_id,
            additional_kwargs=additional_kwargs,
        )

    except Exception as e:
        st.write(f"Something went wrong.\n\n{e}")
        return ToolMessage(
            content=f"Tool was called but failed to get stock prices completely: {e}",
            tool_call_id=tool_id,
            additional_kwargs=additional_kwargs,
        )


def handle_text2audio(
    tool_name: str,
    tool_id: str,
    **kvargs,
) -> ToolMessage:
    additional_kwargs = {}
    try:
        func = FUN_MAPPING.get(tool_name, None)
        if func:
            audio_path = func(**kvargs)
        else:
            raise ValueError("No tool provided.")

        additional_kwargs["audio_path"] = audio_path
        return ToolMessage(
            content=(
                f"Audio has been generated successfully, tool has finished, an audio path: {audio_path}"
                if audio_path and audio_path != ""
                else "Tool called, nothing was responsed by model."
            ),
            tool_call_id=tool_id,
            additional_kwargs=additional_kwargs,
        )
    except Exception as e:
        st.write(f"Something went wrong.\n\n{e}")
        return ToolMessage(
            content=f"Tool was called but failed for:\n\n{e}",
            tool_call_id=tool_id,
            additional_kwargs=additional_kwargs,
        )


def handle_speech2text(
    tool_name: str,
    tool_id: str,
    base64_speech_audio: bytes,
) -> ToolMessage:
    additional_kwargs = {}
    try:
        func = FUN_MAPPING.get(tool_name, None)
        if func:
            text = func(base64_speech_audio)
        else:
            raise ValueError("No tool provided.")

        additional_kwargs["string"] = text
        return ToolMessage(
            content=(
                f"Speech to Text successfully, tool has finished, output text:\n\n{text}"
                if text and text != ""
                else "Tool called, nothing was responsed by model."
            ),
            tool_call_id=tool_id,
            additional_kwargs=additional_kwargs,
        )

    except Exception as e:
        st.write(f"Something went wrong.\n\n{e}")
        return ToolMessage(
            content=f"Tool was called but failed for:\n\n{e}",
            tool_call_id=tool_id,
            additional_kwargs=additional_kwargs,
        )


# ------------------------------------------chat-----------------------------------------------
def tool_call_proc(
    tool_call: Dict,
    base64_object: bytes = None,
) -> ToolMessage:
    tool_id, tool_name = tool_call["id"], tool_call["name"]
    match tool_name:
        case "GenerateImageTool":
            context = tool_call["args"]["context"]
            return handle_generate_image(tool_name, tool_id, context)
        case "AnnotateImageTool":
            image_description = tool_call["args"]["image_description"]
            return handle_annotate_image(
                tool_name, tool_id, base64_object, image_description
            )
        case "RunSearchAgentTool":
            topic = tool_call["args"]["topic"]
            return handle_search_agent(tool_name, tool_id, topic)
        case "Text2SpeechTool":
            text = tool_call["args"]["text"]
            return handle_text2audio(tool_name, tool_id, text=text)
        case "Text2MusicTool":
            prompt = tool_call["args"]["prompt"]
            return handle_text2audio(tool_name, tool_id, prompt=prompt)
        case "SynthesizeAudioTool":
            speech_text = tool_call["args"]["speech_text"]
            prompt = tool_call["args"].get("prompt", speech_text)
            return handle_text2audio(
                tool_name, tool_id, speech_text=speech_text, prompt=prompt
            )
        case "Speech2TextTool":
            return handle_speech2text(tool_name, tool_id, base64_object)
        case "GetCurrentTimeTool":
            return handle_get_current_time(tool_name, tool_id)
        case "GetAndPlotStockPrices":
            now = datetime.now()
            tool_stock_symbols = tool_call["args"]["stock_symbols"]
            tool_start_date = tool_call["args"].get("start_date", now)
            tool_end_date = tool_call["args"].get("end_date", now)
            return handle_get_stock_prices(
                tool_name, tool_id, tool_stock_symbols, tool_start_date, tool_end_date
            )
        case _:
            return ToolMessage(
                content=f"Handle the UNKNOWN tool call: {tool_name}",
                tool_call_id=tool_id,
                additional_kwargs=dict(),
            )


def chat_with_model(
    model: BaseChatModel,
    base64_object: bytes = None,
    media_type: str = None,
    streaming=False,
    image_width=500,
    audio_auto_play=True,
):
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history" not in st.session_state:
        st.session_state.history = ChatMessageHistory()
    for message in st.session_state.messages:
        # pretty_print("message", message)
        if message["role"] == "tool":
            if message["additional_kwargs"].get("image_path"):
                with st.chat_message("assistant"):
                    st.image(
                        message["additional_kwargs"]["image_path"], width=image_width
                    )
            if message["additional_kwargs"].get("audio_path"):
                with st.chat_message("assistant"):
                    st.audio(
                        message["additional_kwargs"]["audio_path"],
                        autoplay=audio_auto_play,
                    )
        else:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if prompt := st.chat_input("Write...", key="chat_input"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                should_continue, tool_messages = True, None
                while should_continue:
                    if not tool_messages:
                        chat_chain = RunnableWithMessageHistory(
                            create_message_chain(
                                model,
                                base64_object,
                                media_type,
                            ),
                            lambda _: st.session_state.history,
                            input_messages_key="query",
                            history_messages_key="history",
                        )
                        call_model = (
                            chat_chain.invoke if not streaming else chat_chain.stream
                        )
                        res = call_model(
                            {"query": prompt},
                            {"configurable": {"session_id": None}},
                        )
                    else:
                        chat_chain = RunnableWithMessageHistory(
                            create_tool_chain(model), lambda _: st.session_state.history
                        )
                        call_model = (
                            chat_chain.invoke if not streaming else chat_chain.stream
                        )
                        res = call_model(
                            {None: tool_messages},
                            {"configurable": {"session_id": None}},
                        )
                        del tool_messages

                    content, additional_kwargs, tool_calls = None, dict(), None
                    if not streaming:
                        content = res.content
                        st.markdown(content)
                    else:
                        content = st.write_stream(res)

                    pretty_print("history", st.session_state.history)
                    last_ai_msg = st.session_state.history.messages[-1]

                    def _has_tool_calls() -> bool:
                        return len(last_ai_msg.tool_calls) > 0

                    async def _run_tool_call_proc(tool_call: Dict) -> ToolMessage:
                        tool_msg = tool_call_proc(tool_call, base64_object)
                        additional_kwargs = tool_msg.additional_kwargs
                        if additional_kwargs.get("image_path"):
                            st.image(
                                additional_kwargs.get("image_path"),
                                width=image_width,
                            )
                        if additional_kwargs.get("audio_path"):
                            st.audio(
                                additional_kwargs.get("audio_path"),
                                autoplay=audio_auto_play,
                            )
                        return tool_msg

                    if _has_tool_calls():
                        streaming, should_continue = False, True
                        tool_calls = last_ai_msg.tool_calls
                        tasks = [
                            _run_tool_call_proc(tool_call) for tool_call in tool_calls
                        ]
                        tool_messages = asyncio.run(asyncio.gather(*tasks))
                    else:
                        should_continue = False

                    if not _has_tool_calls():
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": content,
                                "additional_kwargs": additional_kwargs,
                            }
                        )
                    else:
                        for tool_msg in tool_messages:
                            st.session_state.messages.append(
                                {
                                    "role": "tool",
                                    "content": tool_msg.content,
                                    "additional_kwargs": tool_msg.additional_kwargs,
                                }
                            )


# ------------------------------------------main, app entry------------------------------------------
async def main():
    base64_object, media_type, uploaded = None, None, doc_uploader()
    if uploaded is not None:
        base64_object, media_type = uploaded[0], uploaded[1]
        if media_type == MediaType.IMAGE:
            st.sidebar.image(st.session_state["file_name"], use_column_width=True)
        else:
            st.sidebar.audio(st.session_state["file_name"])
    else:
        pretty_print("uploaded", False)

    st.sidebar.slider("Temperature", 0.0, 1.0, 0.0, key="key_temperature")
    model_sel = st.sidebar.selectbox("Model", ["GPT-4o", "Gemini Pro"], index=0)
    if model_sel == "Gemini Pro":
        used_model = ChatGoogleGenerativeAI(
            model=GOOGLE_LLM,
            temperature=st.session_state.key_temperature,
            max_tokens=MAX_TOKEN,
        )
    else:
        used_model = ChatOpenAI(
            model=OPENAI_LLM,
            temperature=st.session_state.key_temperature,
            max_tokens=MAX_TOKEN,
        )
    ######################## config tool-bindings ########################
    partial_generate_image = partial(
        generate_image,
        model=ChatOpenAI(
            model=OPENAI_LLM,
            temperature=st.session_state.key_temperature,
            max_tokens=MAX_TOKEN,
        ),
    )
    partial_annotate_image = partial(
        annotate_image,
        used_model,
    )
    partial_run_search_agent = partial(
        run_search_agent, create_search_agent(used_model)
    )
    partial_text2speech = partial(
        text2speech, voice_type=st.session_state.get("voice_types", "alloy")
    )
    partial_synthesize_audio = partial(
        synthesize_audio,
        voice_type=st.session_state.get("voice_types", "alloy"),
        duration=st.session_state.get("music_duration", 15),
    )
    partial_text2music = partial(
        text2music, duration=st.session_state.get("music_duration", 15)
    )
    search_agent_tools.extend([partial(load_urls_tool, used_model)])

    FUN_MAPPING["GenerateImageTool"] = partial_generate_image
    FUN_MAPPING["AnnotateImageTool"] = partial_annotate_image
    FUN_MAPPING["RunSearchAgentTool"] = partial_run_search_agent
    FUN_MAPPING["Text2SpeechTool"] = partial_text2speech
    FUN_MAPPING["Text2MusicTool"] = partial_text2music
    FUN_MAPPING["Speech2TextTool"] = speech2text
    FUN_MAPPING["SynthesizeAudioTool"] = partial_synthesize_audio
    FUN_MAPPING["GetCurrentTimeTool"] = get_current_time
    FUN_MAPPING["GetAndPlotStockPrices"] = get_and_plot_stock_prices

    used_model = used_model.bind_tools(
        [
            GenerateImageTool,
            AnnotateImageTool,
            RunSearchAgentTool,
            Text2SpeechTool,
            Text2MusicTool,
            Speech2TextTool,
            SynthesizeAudioTool,
            GetCurrentTimeTool,
            GetAndPlotStockPrices,
        ]
    )
    ########################################################################
    chat_with_model(
        used_model,
        base64_object,
        media_type,
        streaming=st.session_state.get("key_streaming", True),
        image_width=st.session_state.get("key_width", 300),
        audio_auto_play=st.session_state.get("key_audio_auto_play", True),
    )
    st.sidebar.checkbox("Streamming", True, key="key_streaming")
    st.sidebar.slider("Image Width", 100, 1000, 500, 100, key="key_width")
    audio_setting_cols = st.sidebar.columns(2)
    audio_setting_cols[0].checkbox("Audio Auto play", True, key="key_audio_auto_play")
    audio_setting_cols[1].selectbox(
        "Open AI voice types",
        ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
        index=0,
        key="voice_types",
    )
    music_duration = st.sidebar.slider(
        "Music Duration", 10, 30, 15, 1, key="key_music_duration"
    )
    if not os.environ.get("GOOGLE_CSE_ID") or not os.environ.get("GOOGLE_CSE_KEY"):
        st.warning(
            """For Google Search, set GOOGLE_CSE_ID, details: 
key: https://developers.google.com/custom-search/docs/paid_element#api_key
; what: https://support.google.com/programmable-search/answer/12499034?hl=en
; enable: https://console.cloud.google.com/apis/library/customsearch.googleapis
"""
        )


if __name__ == "__main__":
    asyncio.run(main())

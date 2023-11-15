import os
import sys
import time
from typing import Optional

import numpy as np
import streamlit as st
import supervision as sv
import torch
from api.grounding_dino_model import GroundingDINOModel
from dotenv import find_dotenv, load_dotenv
from langchain.agents import initialize_agent
from langchain.agents.agent import AgentExecutor, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.tools import BaseTool
from loguru import logger
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

_ = load_dotenv(find_dotenv())
# os.getenv('OPENAI_API_KEY')

logger.remove()
logger.add(sys.stderr, level="INFO")

st.set_page_config(layout="wide")


class ImageDescriber:
    def __init__(self, model_name: str, device: str) -> None:
        self._device = device

        self._processor = BlipProcessor.from_pretrained(model_name)
        self._model = BlipForConditionalGeneration.from_pretrained(model_name).to(
            device
        )

    def __call__(self, image_path: str) -> str:
        image_obj = Image.open(image_path).convert("RGB")
        inputs = self._processor(image_obj, return_tensors="pt").to(self._device)
        output = self._model.generate(**inputs)
        return self._processor.decode(output[0], skip_special_tokens=True)


class ImageDescriberTool(BaseTool):
    name = "Describe image tool"
    description = "Use this tool to describe found objects an image"

    image_describer: Optional[ImageDescriber] = None

    def setup(self, image_describer: ImageDescriber) -> BaseTool:
        self.image_describer = image_describer
        return self

    def _run(self, image_path: str) -> str:
        return self.image_describer(image_path)

    def _arun(self, query: str):
        raise NotImplementedError


class PromptGeneratorTool(BaseTool):
    name = "Image object detection prompt generator tool"
    description = "Use this tool to generate prompt based on the description of the image for object detection model"

    llm: Optional[BaseLanguageModel] = None

    def setup(self, llm: BaseLanguageModel) -> BaseTool:
        self.llm = llm
        return self

    def _run(self, image_desc: str) -> str:
        logger.debug(f"Image description: {image_desc}")
        input_msg = [
            HumanMessage(
                content=f"""Remove the stop words and useless words, only keep the 'objects', from the following sentence:
                
                {image_desc}
                
                List the objects, separating each with a comma. 
                """
            )
        ]
        # Use simple, fundamental names to describe each object. For instance, use 'tree' instead of 'Christmas tree', or 'girl' instead of 'a little girl'.
        gen_prompt = self.llm(input_msg)
        logger.debug(f"Generated prompt: {gen_prompt}")
        return gen_prompt

    def _arun(self, query: str):
        raise NotImplementedError


class ObjectDetectionTool(BaseTool):
    name = "Object detection on image tool"
    description = "Use this tool to perform an object detection model on an image (read an image path) to detect object with a text prompt"

    groundingDINO_model: Optional[GroundingDINOModel] = None
    output_quality: int = 70

    def setup(
        self, groundingDINO_model: GroundingDINOModel, output_quality=70
    ) -> BaseTool:
        self.groundingDINO_model = groundingDINO_model
        self.output_quality = output_quality
        return self

    def _run(self, image_path: str, prompt: str) -> str:
        logger.debug(f"Image path: {image_path}, prompt: {prompt}")
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        detections, labels = self.groundingDINO_model(
            False, image=image_np, caption=prompt
        )

        image_det_np = image_np
        if len(detections.xyxy) > 0:
            logger.debug(f"detections: {detections}, labels: {labels}")
            box_annotator = sv.BoxAnnotator()
            image_det_np = box_annotator.annotate(
                scene=image_np,
                detections=detections,
                skip_label=False,
                labels=labels,
            )
        output_dir = "output/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        now_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
        output_img_path = os.path.join(output_dir, f"{now_time}.png")
        image = Image.fromarray(image_det_np)
        image.save(
            output_img_path, format="PNG", optimize=True, quality=self.output_quality
        )
        return output_img_path

    def _arun(self, query: str):
        raise NotImplementedError


class App:
    _agent: AgentExecutor

    def __init__(self, device) -> None:
        self.groundingDINO_model = GroundingDINOModel.create_instance(
            device=device,
            groundingDINO_type=st.session_state.get(
                "groundingDINO_model", "swint_ogc"
            ),  # "swint_ogc",  # swinb_cogcoor
        ).setup(
            box_threshold=st.session_state.get("box_threshold", 0.35),
            text_threshold=st.session_state.get("text_threshold", 0.25),
        )
        self.image_describer = ImageDescriber(
            st.session_state.get(
                "blip-image-captioning",
                "Salesforce/blip-image-captioning-base",
            ),  # "Salesforce/blip-image-captioning-base", "Salesforce/blip-image-captioning-large"
            device,
        )

        self.output_quality = st.session_state.get("output_quality", 70)

        if "agent" not in st.session_state:
            llm = (
                ChatOpenAI(model="gpt-4-1106-preview", temperature=0)
                if "llm" not in st.session_state
                else st.session_state["llm"]
            )
            logger.debug("Creating new agent")
            self._agent = initialize_agent(
                agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                tools=[
                    ImageDescriberTool().setup(image_describer=self.image_describer),
                    ObjectDetectionTool().setup(
                        groundingDINO_model=self.groundingDINO_model,
                        output_quality=self.output_quality,
                    ),
                    PromptGeneratorTool().setup(llm),
                ],
                # return_intermediate_steps=True,
                llm=llm,
                verbose=True,
                max_iterations=3,
                early_stopping_method="generate",
                memory=ConversationBufferMemory(
                    memory_key="chat_history",
                    # input_key="input",
                    # output_key="output",
                    return_messages=True,  # then buffer === buffer_as_messages (a list) instead pure str returning
                ),
            )
            st.session_state["agent"] = self._agent
        else:
            logger.debug("Loading existing agent")
            self._agent = st.session_state["agent"]

        # self._structured_output_parser = StructuredOutputParser.from_response_schemas(
        #     [
        #         ResponseSchema(
        #             name="result",
        #             description="""A structured output of the image description and the path of a detection result image:
        #                 { "output": string // where the image was output',  "description": string // the description of the image' }
        #             """,
        #         ),
        #     ]
        # )

    def _abbr(self, msg) -> str:
        if isinstance(msg, HumanMessage):
            return "user"
        elif isinstance(msg, AIMessage):
            return "assistant"
        else:
            raise ValueError(f"Unknown msg type: {msg}")

    def _upload_image(self) -> None:
        with st.sidebar:
            uploaded_image = st.file_uploader("Upload an image")
            if uploaded_image:
                tmp_dir = "tmp/"
                if not os.path.exists(tmp_dir):
                    os.makedirs(tmp_dir)
                temp_file_path = os.path.join(tmp_dir, f"{uploaded_image.name}")
                with open(temp_file_path, "wb") as file:
                    file.write(uploaded_image.getvalue())
                    file_name = uploaded_image.name
                    logger.debug(f"Uploaded {file_name}")
                st.sidebar.image(temp_file_path, width=200)
                self._image_agents_handler(image_path=temp_file_path)
                # os.remove(temp_file_path)

    def _image_agents_handler(self, image_path: str) -> str:
        try:
            result = self._agent(
                f"""Describe the following image:\n{image_path} and detect objects on it with the description as prompt. 
                Only output description and detection result image path. Split the description and image path with ';'.
                """,
            )
            # The format instructions is {self._structured_output_parser.get_format_instructions()}.
            logger.debug(result)
        except Exception as e:
            logger.error(e)

    def run(self) -> None:
        st.title("Image Auto Annotation (auto object detection)")

        self._upload_image()

        st.sidebar.markdown("---")
        st.sidebar.markdown("## blip-image-captioning for image describing")
        st.sidebar.radio(
            "Select a model",
            (
                "Salesforce/blip-image-captioning-base",
                "Salesforce/blip-image-captioning-large",
            ),
            index=0,
            key="blip-image-captioning",
        )

        st.sidebar.markdown("---")
        st.sidebar.markdown("## GroundingDINO for object detection")
        st.sidebar.slider(
            "Box threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.35,
            key="box_threshold",
        )
        st.sidebar.slider(
            "Text threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            key="text_threshold",
        )
        st.sidebar.radio(
            "Select a model",
            ("swint_ogc", "swinb_cogcoor"),
            index=0,
            key="groundingDINO_model",
        )

        st.sidebar.markdown("---")
        st.sidebar.markdown("## Output")
        st.sidebar.slider(
            "Image quality",
            min_value=0,
            max_value=100,
            value=70,
            key="output_quality",
        )

        st.chat_message(name="ai").write(
            "Hey, I can describe an image you uploaded and more."
        )

        logger.debug(self._agent.memory.buffer)
        for idx, msg in enumerate(self._agent.memory.buffer[::-1]):
            if isinstance(msg, AIMessage):
                img_desc_path = msg.content.split(";")
                img_desc, img_path = img_desc_path[0].strip(), img_desc_path[1].strip()
                st.chat_message(name=self._abbr(msg)).write(img_desc)
                st.image(img_path, width=612)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    App(device=device).run()

import base64
import os
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict


import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.llms.ollama import Ollama
from langchain_core.messages import HumanMessage
from langchain.prompts import (
    ChatPromptTemplate,
)
from langchain_core.output_parsers.transform import BaseTransformOutputParser
from langgraph.graph import END, StateGraph
from loguru import logger
from rich.pretty import pprint
from ultralytics import YOLO
from ultralytics.engine.results import Boxes, Results
from PIL import Image
import re

from transformers import BlipForConditionalGeneration, BlipProcessor
import torch

VERBOSE = True


def pretty_print(title: str = None, content: Any = None):
    if not VERBOSE:
        return

    if title is None:
        print(content)
        return
    print(title)
    pprint(content)


class LabelsOutputParser(BaseTransformOutputParser[str]):
    """OutputParser that parses LLMResult into the top likely string."""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this class is serializable."""
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "output_parser"]

    @property
    def _type(self) -> str:
        """Return the output parser type for serialization."""
        return "default"

    def parse(self, text: str) -> str:
        """Returns the input text with no changes."""
        pretty_print("parse", text)
        match = re.search(r'coco_labels = "([^"]+)"', text)
        if match:
            labels = match.group(1).split(",")
        else:
            labels = ""
        return labels


class ImageDescriber:
    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-base",
        device: str = None,
    ) -> None:
        self._device = device
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._processor = BlipProcessor.from_pretrained(model_name)
        self._model = BlipForConditionalGeneration.from_pretrained(model_name).to(
            device
        )

    def __call__(self, image_path: str) -> str:
        image_obj = Image.open(image_path).convert("RGB")
        inputs = self._processor(image_obj, return_tensors="pt").to(self._device)
        output = self._model.generate(max_new_tokens=1024, **inputs)
        return self._processor.decode(output[0], skip_special_tokens=True)


image_describer = ImageDescriber(model_name="Salesforce/blip-image-captioning-base")


class GraphState(TypedDict):
    temperature: Optional[float] = 0.5
    is_local_image_detector: Optional[bool] = True
    uploaded_file_path: Optional[str] = None
    image_description: Optional[str] = None
    labels: Optional[List[str]] = None
    output_image_path: Optional[str] = None


def image_detector(state: GraphState) -> Dict[str, str]:
    """Detects objects in an image as much as possible and returns the description of the image."""
    if state.get("is_local_image_detector"):
        image_description = image_describer(state.get("uploaded_file_path"))
        return {"image_description": image_description}

    with open(state.get("uploaded_file_path"), "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        # pretty_print("base64_image", base64_image)
        prompt = ChatPromptTemplate.from_messages(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": "What is this image showing"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ]
                )
            ]
        )
        chat = ChatOpenAI(
            model="gpt-4-vision-preview",
            temperature=float(state.get("temperature")),
            max_tokens=1024 * 2,
        )
        image_description: str = (prompt | chat | StrOutputParser()).invoke(
            {"base64_image": base64_image}
        )
        return {"image_description": image_description}


def coco_label_extractor(state: GraphState) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """As a helpful assistant you should extract COCO compatible labels from a image description. 
                Try to extract labels as much as possible from the description and convert to COCO compatible labels.
                ONLY return the labels as a comma separated string without any instruction information,special characters,mathamatical symbols,or any other information.
                """,
            ),
            ("human", "{img_desc}"),
        ]
    )
    llm = Ollama(
        base_url="http://localhost:11434",
        model="gemma:2b-instruct",
        temperature=0,
    )

    labels: str = (prompt | llm | LabelsOutputParser()).invoke(
        {"img_desc": state.get("image_description")}
    )
    return {"labels": labels}


def annotate_image(state: GraphState) -> Results:
    """Annotate an image with classes (COCO dataset types)"""
    model = YOLO("yolov8s-world.pt")  # or select yolov8m/l-world.pt for different sizes

    classes = state.get("labels")
    pretty_print("classes", classes)
    if len(classes) > 0:
        model.set_classes(classes=classes)

    preds = model.predict(state.get("uploaded_file_path"))
    results: Results = preds[0]
    output_image_path: str = results.save("tmp/langgraph_auto_annotation_res.jpeg")
    return {"output_image_path": output_image_path}


def continue_next(
    state: GraphState,
) -> Literal["to_coco_label_extractor", "to_annotate_image"]:
    if state.get("image_description") is not None and state.get("labels") is None:
        return "to_coco_label_extractor"

    return "to_annotate_image"


def create_graph() -> StateGraph:
    graph = StateGraph(GraphState)
    graph.add_node("image_detector", image_detector)
    graph.add_node("coco_label_extractor", coco_label_extractor)
    graph.add_node("annotate_image", annotate_image)

    graph.set_entry_point("image_detector")
    graph.add_edge("coco_label_extractor", "annotate_image")
    graph.add_edge("annotate_image", END)

    graph.add_conditional_edges(
        "image_detector",
        continue_next,
        {
            "to_coco_label_extractor": "coco_label_extractor",
            "to_annotate_image": "annotate_image",
        },
    )

    return graph


def doc_uploader() -> StateGraph | None:
    with st.sidebar:
        uploaded_doc = st.file_uploader("# Upload an image", key="doc_uploader")
        if not uploaded_doc:
            st.session_state["file_name"] = None
            st.session_state["queries"] = None
            logger.debug("No file uploaded")
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

                # os.remove(temp_file_path)

                if st.session_state.get("file_name") == file_name:
                    logger.debug("Same file, same quiries")
                    return st.session_state["queries"]

                logger.debug("New file, new queries")
                st.session_state["file_name"] = temp_file_path
                st.session_state["graph"] = create_graph()
                st.sidebar.image(Image.open(temp_file_path))
                return st.session_state["graph"]
        return None


def main():
    if not bool(
        st.sidebar.selectbox(
            "True for BLIP (full local), False for OpenAI GPT-4 Vision (remote)",
            [True, False],
            key="is_local_image_detector",
        )
    ):
        st.sidebar.slider("Temperature", 0.0, 1.0, 1.0, key="temperature")
    else:
        st.session_state["temperature"] = 0.5

    graph: StateGraph | None = doc_uploader()
    if graph is None:
        return
    app = graph.compile()
    pretty_print("temperture", st.session_state["temperature"])
    result = app.invoke(
        {
            "uploaded_file_path": st.session_state["file_name"],
            "temperature": float(st.session_state["temperature"]),
            "is_local_image_detector": bool(
                st.session_state["is_local_image_detector"]
            ),
        }
    )
    pretty_print("result", result)

    cols = st.columns([1, 1])
    with cols[0]:
        st.image(Image.open(result["output_image_path"]))
    with cols[1]:
        st.write("### Image Description")
        st.write(result["image_description"])
        st.write("### Labels")
        st.write(result["labels"])


if __name__ == "__main__":
    main()

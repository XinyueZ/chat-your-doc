# %% Helper functions ##########################################################
from typing import List
from rich.pretty import pprint


def pretty_print(title: str = None, content: str = None):
    if title is None:
        print(content)
        return
    print(title)
    pprint(content)


# %% Customize parser ##########################################################
from langchain_core.output_parsers.transform import BaseTransformOutputParser


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
        match = re.search(r'coco_labels = "([^"]+)"', text)
        if match:
            labels = match.group(1).split(",")
        else:
            labels = ""
        return labels


# %% Annotate image ############################################################
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.engine.results import Results

import matplotlib.pyplot as plt


def annotate_image(
    img_path: str, classes: list[str] = None, model_ckp: str = "yolov8s-world.pt"
) -> Results:
    """Annotate an image with classes (COCO dataset types)"""
    model = YOLO(model_ckp)  # or select yolov8m/l-world.pt for different sizes

    if classes is not None:
        model.set_classes(classes)

    preds = model.predict(img_path)
    results: Results = preds[0]
    # results.show()
    # pretty_print(content=results)
    return results


res: Results = annotate_image(classes=["person", "tree"], img_path="tmp/xmasroom.jpeg")
save_1 = res.save("tmp/xmasroom_annotated_1.jpeg")

res: Results = annotate_image(img_path="tmp/xmasroom.jpeg")
save_2 = res.save("tmp/xmasroom_annotated_2.jpeg")


def show_image(img_path_list: list[str], titles: list[str]):
    for i, (path, title) in enumerate(zip(img_path_list, titles)):
        axs = plt.subplot(1, len(img_path_list), i + 1)
        axs.set_title(title)
        axs.axis("off")
        plt.imshow(plt.imread(path))


show_image([save_1, save_2], ["annotated", "auto-annotated"])

# %% Image Detector ###########################################################
from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOpenAI

from langchain.schema.output_parser import StrOutputParser
import base64


def image_detector(image_path: str) -> str:
    """Detects objects in an image as much as possible and returns the description of the image."""
    with open(image_path, "rb") as image_file:
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
            model="gpt-4-vision-preview", temperature=0, max_tokens=1024 * 2
        )
        return (prompt | chat | StrOutputParser()).invoke(
            {"base64_image": base64_image}
        )


image_description = image_detector("tmp/xmasroom.jpeg")
pretty_print("image_description", image_description)

# %% COCO label extractor #####################################################
from langchain.prompts.chat import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain.schema.output_parser import StrOutputParser


def coco_label_extractor(img_desc: str) -> List[str]:
    """Read an image description and extract COCO defined labels as much as possible from the description."""
    chat_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """As a helpful assistant you should extract COCO and ImageNet compatible labels from a image description. 
                Try to extract labels as much as possible from the description and convert to COCO and ImageNet compatible labels.
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
        top_k=3,
    )

    return (chat_template | llm | LabelsOutputParser()).invoke({"img_desc": img_desc})


coco_labels = coco_label_extractor(image_description)
pretty_print("coco_labels", coco_labels)

# %% Process image annotation ################################################
res: Results = annotate_image(classes=coco_labels, img_path="tmp/xmasroom.jpeg")
save_1 = res.save("tmp/xmasroom_annotated_1.jpeg")

res: Results = annotate_image(img_path="tmp/xmasroom.jpeg")
save_2 = res.save("tmp/xmasroom_annotated_2.jpeg")

show_image([save_1, save_2], ["YoLoV8 World + OpenAI Vision", "YoLoV8 World"])

# %% FewShotChatMessagePromptTemplate ##########################################
from langchain.prompts import (
    PromptTemplate,
    FewShotChatMessagePromptTemplate,
    ChatPromptTemplate,
    FewShotPromptTemplate,
)

examples = [
    {
        "img_desc": """The image shows two individuals, an adult and a child, standing on a sandy beach near 
            the water's edge. The adult appears to be looking down 
            towards the child, who is facing them. The child is wearing a pink dress and 
            the adult is dressed in dark clothing. 
            It looks like a peaceful scene, possibly a family moment, with the calm water 
            in the background and a gentle interaction between the two.""",
        "labels": """adult,Child,pink dress,dark clothing""",
    },
    {
        "img_desc": """The image shows two individuals, an adult and a child, 
            standing on a sandy beach near the water's edge. The adult appears to be looking down 
            towards the child, who is wearing a pink dress. The water is calm, and there are small waves 
            lapping at the shore. The lighting suggests it could be late afternoon or early evening, given
            the long shadows cast on the sand. It's a tranquil scene that could be indicative of a leisurely day spent at the beach.""",
        "labels": """child,beach,water,sand""",
    },
    {
        "img_desc": """The image shows two individuals, an adult and a child, standing on a sandy beach near the water's edge. 
            The adult appears to be looking down towards the child, who is facing them. 
            The child is wearing a pink dress and the adult is dressed in dark clothing. 
            It looks like a calm moment, possibly a conversation or a shared observation of the surroundings. 
            The water is calm with small waves lapping at the shore, and the lighting suggests it could be late afternoon or early evening.""",
        "labels": """adult,child,beach,water,sun,light""",
    },
]
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{img_desc}"),
        ("ai", "{labels}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)
final_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Extract COCO defined labels from the image description.
            Provide the result ONLY as a simple list separated by commas.
            """,
        ),
        few_shot_prompt,
        ("human", "{img_desc}"),
    ]
)
print(
    final_prompt.format(
        img_desc="The image shows two individuals, an adult and a child, standing on a sandy beach near the water's edge. The adult appears to be looking down towards the child, who is facing them. The child is wearing a pink dress and the adult is dressed in dark clothing. It looks like a peaceful scene, possibly a family moment, with the calm water in the background and a gentle interaction between the two."
    )
)

# %% Try gemma vision possible (sad, it cannot) ##########################################################
with open("tmp/xmasroom.jpeg", "rb") as image_file:
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
    chat = Ollama(
        base_url="http://localhost:11434",
        model="gemma:2b-instruct",
        temperature=0,
        top_k=3,
    )
    res = (prompt | chat | StrOutputParser()).invoke({"base64_image": base64_image})
    pretty_print("res", res)

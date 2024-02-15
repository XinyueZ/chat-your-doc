# %% Helper functions
from rich.pretty import pprint


def pretty_print(title: str = None, content: str = None):
    if title is None:
        print(content)
        return
    print(title)
    pprint(content)


# %% Annotate image
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

# %% Image Detector
from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOpenAI

import base64


def image_detector(image_path: str) -> str:
    """Detects objects in an image as much as possible and returns the description of the image."""
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        # pretty_print("base64_image", base64_image)

        chat = ChatOpenAI(
            model="gpt-4-vision-preview", temperature=0, max_tokens=1024 * 2
        )
        res = chat.invoke(
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

        return res.content


image_description = image_detector("tmp/xmasroom.jpeg")
pretty_print("image_description", image_description)

# %% COCO label extractor
from langchain.prompts.chat import ChatPromptTemplate


def coco_label_extractor(img_desc: str) -> str:
    """Read an image description and extract COCO defined labels as much as possible from the description."""
    chat_template = ChatPromptTemplate.from_messages(
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
    human_input = chat_template.format_messages(img_desc=img_desc)
    chat = ChatOpenAI(model="gpt-4-vision-preview", temperature=0, max_tokens=1024)
    return chat.invoke(human_input).content


coco_labels = coco_label_extractor(image_description)
pretty_print("coco_labels", coco_labels)

# %% Process image annotation
res: Results = annotate_image(
    classes=coco_labels.split(","), img_path="tmp/xmasroom.jpeg"
)
save_1 = res.save("tmp/xmasroom_annotated_1.jpeg")

res: Results = annotate_image(img_path="tmp/xmasroom.jpeg")
save_2 = res.save("tmp/xmasroom_annotated_2.jpeg")

show_image([save_1, save_2], ["YoLoV8 World + OpenAI Vision", "YoLoV8 World"])

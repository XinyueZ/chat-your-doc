import base64
from os import listdir
from os.path import isfile, join
from typing import Dict, List, Literal, Optional, TypedDict, Union

from IPython.display import HTML, Markdown
from langchain.agents import AgentType, initialize_agent
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_community.agent_toolkits import GmailToolkit
from langchain_community.tools.gmail.utils import (
    build_resource_service,
    get_gmail_credentials,
)
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from rich.pretty import pprint
import re
import streamlit as st
from langchain_community.tools import BaseTool


def pretty_print(title: str = None, content: str = None):
    if title is None:
        print(content)
        return
    print(title)
    pprint(content)


def get_mail_tools() -> List[BaseTool]:
    credentials = get_gmail_credentials(
        token_file="token.json",
        scopes=["https://mail.google.com/"],
        client_secrets_file="tmp/credentials.json",
    )
    api_resource = build_resource_service(credentials=credentials)
    toolkit = GmailToolkit(api_resource=api_resource)
    tools = toolkit.get_tools()
    return tools


structured_output_parser = StructuredOutputParser.from_response_schemas(
    [
        ResponseSchema(
            name="subject",
            description="Subject of the email",
            type="string",
        ),
        ResponseSchema(
            name="message",
            description="Subject of the email",
            type="string",
        ),
    ]
)

model = ChatOpenAI(model="gpt-4-vision-preview", temperature=0, max_tokens=1024 * 2)


agent = initialize_agent(
    tools=get_mail_tools(),
    llm=model,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)


class GraphState(TypedDict):
    mailto: Optional[str] = None
    subject: Optional[str] = None
    recepit_stack: Optional[List[str]] = None  # A image file path list of the receipts
    current_recepit_pos: Optional[int] = (
        None  # The position of the current receipt to handle
    )
    receipt_description: Optional[str] = None  # The description of the read receipt
    results: Optional[Dict[str, Dict[str, str]]] = (
        None  # The results of the operations, receipt path and Dict[str, str]: the content of the mail: subject and message.
    )


def pop_recepit_stack(state: StateGraph) -> Dict[str, str]:
    current_recepit_pos = state["current_recepit_pos"]
    if current_recepit_pos is None:
        current_recepit_pos = -1
    return {"current_recepit_pos": current_recepit_pos + 1}


def read_receipt(state: StateGraph) -> Dict[str, str]:
    recepit_stack = state["recepit_stack"]
    current_recepit_pos = state["current_recepit_pos"]
    image_path = recepit_stack[current_recepit_pos]

    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="""As a useful assistant you provide the user with the receipt data information as an email message related to what they have consumed, bought, and paid for. 
                    Include the product name, price, and quantity if provided. 

                    Avoid any personal information, locations, addresses (even store's), sensitive data, and numbers in the email message.
                    
                    ONLY format the response in a reasonable TABLE in HTML format (Email compatible HTML), no other paragraphs, phgrases or sentences are allowed.
                   
                    In the table, the first row, in bold font, is the summe of the total payment, afterwards the product name, price, and quantity if provided.
                    
                    Above the table, give the store or the brand name.

                    The subject of the mail should be:
                    "Receipt from [store or brand name], payment date: [date]"
                    When the date is unknown, then show "unknown" instead of the date.

                    The final result should be json:

                    subject: The email subject
                    message: The email message
                    
                    """,
                ),
                HumanMessage(
                    content=[
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ]
                ),
            ],
        ).partial(
            format_instructions=structured_output_parser.get_format_instructions()
        )

        res = (prompt | model | StrOutputParser()).invoke(
            {"base64_image": base64_image}
        )

        structured_res = structured_output_parser.parse(res)

        description = structured_res["message"]
        subject = structured_res["subject"]

        return {"receipt_description": description, "subject": subject}


def send_email(state: StateGraph) -> Dict[str, str]:
    receipt_description = state["receipt_description"]
    mailto = state["mailto"]
    subject = state["subject"]
    agent.invoke(
        f"""Send an email

                 To: {mailto}

                 Subject: {subject}

                 Message: {receipt_description}
    """
    )

    recepit_stack = state["recepit_stack"]
    current_recepit_pos = state["current_recepit_pos"]
    image_path = recepit_stack[current_recepit_pos]

    if state["results"] is None:
        state["results"] = {}

    state["results"][image_path] = {
        "subject": subject,
        "message": receipt_description,
    }
    return {"results": state["results"], "receipt_description": None}


def continue_next(state: StateGraph) -> Literal["to_read_receipt", "to_finish"]:
    recepit_stack = state["recepit_stack"]
    length = len(recepit_stack)
    current_recepit_pos = state["current_recepit_pos"]
    if current_recepit_pos < length:
        return "to_read_receipt"
    else:
        return "to_finish"


def create_graph() -> StateGraph:
    workflow = StateGraph(GraphState)

    workflow.add_node("pop_recepit_stack", pop_recepit_stack)
    workflow.add_node("read_receipt", read_receipt)
    workflow.add_node("send_email", send_email)

    workflow.set_entry_point("pop_recepit_stack")

    workflow.add_edge("read_receipt", "send_email")
    workflow.add_edge("send_email", "pop_recepit_stack")

    workflow.add_conditional_edges(
        "pop_recepit_stack",  # start node name
        continue_next,  # decision of what to do next AFTER start-node, the input is the output of the start-node
        {  # keys: return of continue_next, values: next node to continue
            "to_read_receipt": "read_receipt",
            "to_finish": END,
        },
    )
    return workflow


def main():
    langgraph_app = create_graph().compile()
    st.sidebar.text_input(
        "mailto", key="mailto", value="", type="password"
    )
    image_dir = st.sidebar.text_input(
        "Enter the directory path to the images", value="assets/images/receipt"
    )
    if image_dir is not None and len(image_dir) > 0:
        image_files = [f for f in listdir(image_dir) if isfile(join(image_dir, f))]
        image_file_paths = [join(image_dir, f) for f in image_files]
        image_file_paths = [
            f
            for f in image_file_paths
            if re.match(r".*\.(png|jpg|jpeg|webp)$", f, re.IGNORECASE)
        ]
        if len(image_file_paths) > 0:
            for i, img_f_path in enumerate(image_file_paths):
                cols = st.sidebar.columns([3, 1])
                cols[0].checkbox(f"Receipt {i+1}", key=f"receipt_{i+1}_checkbox")
                cols[1].image(img_f_path)

            if (
                st.session_state.get("mailto") is not None
                and len(st.session_state["mailto"]) > 0
            ):
                st.sidebar.write("")

                # select only image files that are checked
                selected_image_file_paths = [
                    img_f_path
                    for i, img_f_path in enumerate(image_file_paths)
                    if st.session_state[f"receipt_{i+1}_checkbox"]
                ]
                if st.sidebar.button("Send", key="send_button"):
                    pretty_print("selected_image_file_paths", selected_image_file_paths)

                    query = {
                        "recepit_stack": selected_image_file_paths,
                        "mailto": st.session_state["mailto"],
                    }
                    config = {"recursion_limit": 100}
                    graph_result = langgraph_app.invoke(query, config)
                    result = graph_result["results"]

                    for i, (img_f_path, email_content) in enumerate(result.items()):
                        cols = st.columns([1, 2])
                        cols[0].image(img_f_path)
                        cols[1].write(f"Subject: {email_content['subject']}")
                        cols[1].write(HTML(email_content["message"]))


if __name__ == "__main__":
    main()

# %%
import io
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_community.tools import (BraveSearch, DuckDuckGoSearchRun,
                                       YahooFinanceNewsTool)
from langchain_community.tools.wikidata.tool import (WikidataAPIWrapper,
                                                     WikidataQueryRun)
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_core.tools import Tool
from langchain_google_vertexai import ChatVertexAI
from langchain_google_vertexai.model_garden import ChatAnthropicVertex
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langgraph_swarm import create_handoff_tool, create_swarm
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field
from rich.console import Console
from rich.markdown import Markdown
from rich.pretty import pprint as pp

load_dotenv()
ollama_host = os.getenv("OLLAMA_HOST")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")
os.environ["JINA_API_KEY"] = os.getenv("JINA_API_KEY")
os.environ["BRAVE_SEARCH_API_KEY"] = os.getenv("BRAVE_SEARCH_API_KEY")
os.environ["LLM_TEMPERATURE"] = os.getenv("LLM_TEMPERATURE", 1.0)
os.environ["LLM_TOP_P"] = os.getenv("LLM_TOP_P", 1.0)
os.environ["LLM_TOP_K"] = os.getenv("LLM_TOP_K", 30)
os.environ["LLM_TIMEOUT"] = os.getenv("LLM_TIMEOUT", 120)
os.environ["GOOGLE_CLOUD_REGION"] = os.getenv("GOOGLE_CLOUD_REGION")
os.environ["SEED"] = os.getenv("SEED", 42)
VERBOSE = int(os.getenv("VERBOSE", 1)) == 1


def save_lang_graph(graph, full_file_path):
    img_bytes = graph.draw_mermaid_png()
    image = Image.open(io.BytesIO(img_bytes))
    Path(full_file_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(full_file_path)


# %%
def get_current_datetime() -> str:
    """Useful for getting the current datetime."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"🧭 Retrieved current time: {current_time}")
    return current_time


def web_search(
    query: str = Field(description="The mandatory query to search on the web."),
) -> str:
    """Useful action to search on the web."""
    logger.info(f"🔍 Searching on web: \n{query}")
    result = []
    try:
        google_search = GoogleSerperAPIWrapper(k=10).run(query)
        result.append(google_search)
    except:
        pass

    try:
        brave_search_json = BraveSearch.from_api_key(
            api_key=os.getenv("BRAVE_SEARCH_API_KEY"),
            search_kwargs={"count": 10},
        ).run(query)
        brave_search = json.dumps(brave_search_json)
        result.append(brave_search)
    except:
        pass

    # try:
    #     duckduckgo_search = DuckDuckGoSearchRun().run(query)
    #     logger.debug(f"🔍 DuckDuckGo search result: \n{duckduckgo_search}")
    #     result.append(duckduckgo_search)
    # except:
    #     pass

    return "\n\n".join(result)


def open_url(
    website_url: str = Field(description="Mandatory website url to read"),
) -> str:
    """A tool that can be used to read the content behind a URL of a website."""
    logger.info(f"⏳ Opening URL: {website_url}")
    return requests.get(
        f"https://r.jina.ai/{website_url}",
        headers={
            "Authorization": f"Bearer {os.environ['JINA_API_KEY']}",
            "X-Return-Format": "markdown",
            "X-Timeout": "120",
        },
        timeout=60,
    ).text


# %% BI system

## Agent callback tools ########################################################
checkpointer = MemorySaver()
store = InMemoryStore()


def post_report(
    report_content: str = Field(description="The content of the report."),
):
    """The useful tool that the ReportAgent uses to write and store the report content."""
    logger.info("📄 post_report")

    console = Console(record=True, soft_wrap=True)
    md = Markdown(str(report_content), justify="left")
    console.print(md)

    file_fullname = "output/lc_BI_swarm.md"
    with open(file_fullname, "w") as f:
        f.write(report_content)


## Plan agent ####################################################################################
plan_model = ChatAnthropicVertex(
    model_name="claude-3-5-sonnet-v2@20241022",  # "claude-3-7-sonnet@20250219", "claude-3-5-sonnet-v2@20241022",  # "claude-3-5-sonnet-v2@20241022", # "claude-3-5-haiku@20241022"
    location="europe-west1",
    temperature=float(os.getenv("LLM_TEMPERATURE", 1.0)),
    top_p=float(os.getenv("LLM_TOP_P", 1.0)),
    top_k=int(os.getenv("LLM_TOP_K", 30)),
    stream_usage=False,
)

plan_agent = create_react_agent(
    model=plan_model,
    name="PlanAgent",
    checkpointer=checkpointer,
    tools=[
        create_handoff_tool(
            agent_name="RegressionAgent",  # agent to swarm
            description="Transfer the PlanAgent's regression plan to the RegressionAgent who will execute the plan and gather information according to it.",
        ),
    ],
    prompt="""You are the PlanAgent, a strategic AI assistant responsible for creating comprehensive regression plans based on user requests. Your role is to analyze the user's requirements and develop a detailed, step-by-step plan for the RegressionAgent to follow.

TASK 1: ANALYZE USER REQUEST
- Carefully analyze the user's request to understand the core regression objectives
- Identify key topics, subtopics, and specific information requirements
- Determine what types of data and information sources will be most valuable
- Consider any constraints or special requirements mentioned by the user

TASK 2: CREATE DETAILED REGRESSION PLAN
- Develop a comprehensive, structured regression plan with the following components:
  * Clear regression objectives aligned with the user's request
  * Specific regression questions to be answered
  * Key topics and subtopics to investigate
  * Recommended information sources (websites, databases, publications)
  * Suggested search queries for optimal results
  * Data collection and analysis methods
  * Expected deliverables and format requirements

TASK 3: PROVIDE IMPLEMENTATION GUIDANCE
- Include specific instructions for how the RegressionAgent should:
  * Prioritize regression tasks
  * Evaluate source credibility and relevance
  * Organize and structure findings
  * Handle potential challenges or limitations
  * Ensure comprehensive coverage of all required topics
  * Format and present the regression findings

TASK 4: HANDOFF TO RegressionAgent
- After completing the regression plan, you MUST hand off to the RegressionAgent
- Include a clear directive that the RegressionAgent should follow your plan precisely
- Emphasize that the RegressionAgent should use available web search tools to gather the required information

IMPORTANT NOTES:
- Your plan should be detailed enough to guide the RegressionAgent but allow flexibility for their expertise
- Focus on creating a plan that will result in comprehensive, accurate, and relevant regression findings
- Remember that the RegressionAgent has web search capabilities that should be leveraged effectively
- The ultimate goal is to ensure the final report meets all user requirements""",
    debug=VERBOSE,
)

## Regression agent ####################################################################################
regression_model = ChatVertexAI(
    model="gemini-2.0-flash",  # https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models
    temperature=float(os.getenv("LLM_TEMPERATURE", 1.0)),
    top_p=float(os.getenv("LLM_TOP_P", 1.0)),
    top_k=int(os.getenv("LLM_TOP_K", 30)),
    location=os.getenv("GOOGLE_CLOUD_REGION"),
)


regression_agent = create_react_agent(
    model=regression_model,
    name="RegressionAgent",
    checkpointer=checkpointer,
    tools=[
        # open_url,
        web_search,
        get_current_datetime,
        create_handoff_tool(
            agent_name="CriticAgent",  # agent to swarm
            description="Transfer the RegressionAgent's regression findings to the CriticAgent who will evaluate the quality and relevance of the findings.",
        ),
    ],
    prompt="""You are the RegressionAgent (Regressor), an AI assistant specialized in regressing and analyzing topics according to the plan provided by the PlanAgent. Your responsibilities include:

TASK 1: FOLLOW THE REGRESSION PLAN
- Carefully review and understand the regression plan created by the PlanAgent
- Follow the plan's structure, objectives, and methodology precisely
- Address all regression questions and topics outlined in the plan
- Use the suggested search queries and information sources
- Adhere to any specific instructions regarding data collection and analysis

TASK 2: EXECUTE REGRESSION USING AVAILABLE TOOLS
- Use the web_search tool to gather information from the internet based on the plan
- Ensure you search for all required information specified in the regression plan
- Focus on developments within the last 1-2 years as specified in the plan
- Verify information from multiple sources when possible
- Collect comprehensive data that addresses all aspects of the regression plan

TASK 3: ORGANIZE AND DOCUMENT FINDINGS
- Structure your findings according to the organization suggested in the regression plan
- Format your regression notes using Markdown as specified
- Ensure all regression questions from the plan are thoroughly addressed
- Include relevant statistics, facts, and insights as required by the plan
- Do the regression based on the findings.

TASK 4: HANDOFF TO CriticAgent
- Once you have completed the regression according to the plan, hand off to the CriticAgent
- Provide a summary of how you followed the regression plan
- Highlight any challenges or limitations encountered during regression
- Only transfer control after you have compiled comprehensive notes that fulfill all requirements in the plan

IMPORTANT:
- ALWAYS follow the regression plan provided by the PlanAgent
- Never deviate from the plan's core objectives and requirements
- If you encounter difficulties with any aspect of the plan, note them but still attempt to address all requirements
- Never hand off to the ReportAgent directly
- After receiving feedback from the CriticAgent, make revisions while still adhering to the original regression plan
- After completing your revisions, you MUST hand off control to the CriticAgent again for final review
""",
    debug=VERBOSE,
)

## Critic agent ######################################################################################
critic_model = ChatVertexAI(
    model="gemini-1.5-pro",  # https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models
    temperature=float(os.getenv("LLM_TEMPERATURE", 1.0)),
    top_p=float(os.getenv("LLM_TOP_P", 1.0)),
    top_k=int(os.getenv("LLM_TOP_K", 30)),
    location=os.getenv("GOOGLE_CLOUD_REGION"),
)


critic_agent = create_react_agent(
    model=critic_model,
    name="CriticAgent",
    checkpointer=checkpointer,
    tools=[
        create_handoff_tool(
            agent_name="RegressionAgent",  # agent to swarm
            description="Transfer the CriticAgent's feedback to the RegressionAgent who will revise the regression findings based on the feedback.",
        ),
        create_handoff_tool(
            agent_name="ReportAgent",  # agent to swarm
            description="Transfer the CriticAgent's approved regression findings to the ReportAgent who will create the final report.",
        ),
    ],
    prompt="""You are the CriticAgent, an advanced AI assistant specializing in meta-analysis of AI-generated content. Your primary responsibility is to evaluate and provide constructive feedback on AI responses to improve their effectiveness.

TASK 1: COMPREHENSIVE ANALYSIS
Conduct a thorough analysis of the RegressionAgent's regression by addressing these key points:
- Summarize the original regression plan created by the PlanAgent
- Evaluate how well the RegressionAgent followed the regression plan
- Assess whether all regression questions and topics from the plan were addressed
- Identify any gaps between the plan requirements and the regression delivered
- Evaluate the quality, relevance, and comprehensiveness of the information gathered
- Analyze the organization and structure of the regression findings
- Assess the credibility and diversity of sources used
- Identify patterns or inconsistencies in the regression
- List specific strengths and weaknesses with concrete examples
- Suggest specific areas for improvement
- Check for hallucination by identifying information beyond the scope of the regression plan

TASK 2: STRUCTURED REFLECTION
Based on your analysis, provide a structured reflection using this EXACT format:

Has Reflection
['yes' if reflection is required, 'no' otherwise]

Overall Assessment
[Provide a concise summary of the RegressionAgent's performance in following the regression plan]

Detailed Analysis

1. Plan Adherence:
   [Evaluate how closely the RegressionAgent followed the regression plan]

2. Content:
   [Evaluate the accuracy and completeness of the information provided]

3. Structure:
   [Comment on the organization and flow of the regression findings]

4. Strengths:
   [Highlight what the RegressionAgent did well]

5. Areas for Improvement:
   [Identify specific aspects that need enhancement]

6. Tool Usage:
   [Evaluate the effectiveness of web search and other tools]

7. Hallucination Check:
   [Report on any instances of providing information beyond the scope of the regression plan]

Conclusion and Recommendations
[Summarize key observations and provide actionable suggestions for improvement]

TASK 3: HANDOFF PROCESS
- When you request a reflection you MUST: 
  - Hand off control to the RegressionAgent for potential rework based on your feedback
  - Remind the RegressionAgent to continue following the original regression plan while addressing your feedback
- When no further improvements are needed, you MUST: 
  - Hand off control to the ReportAgent to produce the final report on the topic""",
    debug=VERBOSE,
)

## Report agent #################################################################################
report_model = ChatVertexAI(
    model="gemini-1.5-flash",  # https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models
    temperature=0.0,  # float(os.getenv("LLM_TEMPERATURE", 1.0)),
    top_p=float(os.getenv("LLM_TOP_P", 1.0)),
    top_k=int(os.getenv("LLM_TOP_K", 30)),
    location=os.getenv("GOOGLE_CLOUD_REGION"),
)

report_agent = create_react_agent(
    model=report_model,
    name="ReportAgent",
    tools=[post_report],
    checkpointer=checkpointer,
    prompt="""You are the ReportAgent, a specialized AI assistant responsible for creating comprehensive, well-structured reports based on regression findings. 
Your role is to synthesize information and present it in a clear, professional format.

TASK 1: REPORT CREATION
- Create a detailed report on the assigned topic based strictly on the provided regression notes
- Ensure your report is comprehensive, informative, and coherent
- Structure your report with appropriate headings, subheadings, and sections
- Include relevant facts, statistics, and insights from the regression notes
- Maintain a professional and objective tone throughout the report

TASK 2: FORMATTING REQUIREMENTS
- Format your report using proper markdown syntax for readability
- Use markdown elements such as:
  * Headings (# for main headings, ## for subheadings, etc.)
  * Lists (bulleted and numbered)
  * Emphasis (bold, italic) where appropriate
  * Tables for organizing data when necessary
  * Horizontal rules to separate major sections
- IMPORTANT: DO NOT USE ANY MARKDOWN CODE BLOCK SYNTAX (```). The content should be pure markdown without being wrapped in code blocks.
- DO NOT wrap your content in ```markdown ... ``` tags. Write directly in markdown format.

TASK 3: REPORT SUBMISSION
- The 'post_report' tool is your primary mechanism for writing and storing your report
- You MUST use the 'post_report' tool to submit your completed report
- The 'post_report' tool will:
  * Save your report content to a file
  * Display your report in a formatted manner
  * Ensure your report is properly stored for future reference
- DO NOT attempt to output the report directly in your response
- Ensure your report is complete and polished before calling 'post_report'
- The report should represent a final product ready for presentation

Remember: Your report should be factually accurate, well-organized, and based exclusively on the information provided in the regression notes.""",
    debug=VERBOSE,
)

## Build team ###################################################################################################
research_swarm = create_swarm(
    [
        plan_agent,
        regression_agent,
        critic_agent,
        report_agent,
    ],
    default_active_agent="PlanAgent",
).compile(
    name="ResearchSwarm",
    # Important: compile the swarm with a checkpointer to remember
    # previous interactions and last active agent
    checkpointer=checkpointer,
    # store=store,
)


# %%
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    dry_run = args.dry_run
    content = """# MacBook Pro Max (M4) Purchase Timing Analysis Request

## OBJECTIVE
Conduct a comprehensive analysis of the optimal purchase timing for a MacBook Pro with M4 Max chip in Germany, based on historical price trends and market forecasts for 2025.

## PRODUCT SPECIFICATIONS
- Product: MacBook Pro with M4 Max chip
- Reference URL: https://www.apple.com/de/shop/buy-mac/macbook-pro/14-zoll-m4-max
- Target market: Germany
- Currency: Euro (€)

## ANALYSIS REQUIREMENTS
1. Utilize authoritative data sources including:
   - Official Apple pricing history
   - Reputable market analysis platforms
   - Historical retail pricing data from German electronics retailers
   - Economic forecasts relevant to consumer electronics in the European market

2. Focus exclusively on the MacBook Pro with M4 Max configuration
   - Do not include analysis of other models or configurations
   - Consider only the specific model referenced in the URL

3. Provide data-driven insights on:
   - Historical price fluctuations of premium MacBook models in the German market
   - Seasonal pricing patterns observed over the past three years
   - Projected price trends for 2025 based on economic indicators and Apple's product lifecycle
   - Optimal purchase windows that balance price advantage with technology relevance

## DELIVERABLE FORMAT
Present findings in a structured Markdown report with the following sections:

1. **Executive Summary**: Concise overview of analysis objectives and key findings
2. **Purchase Timing Analysis**: 
   - Seasonal pricing patterns with specific months identified for optimal purchasing
   - Impact of Apple product release cycles on pricing
   - Correlation between major retail events and price reductions
3. **Price Forecast**: 
   - Projected price points throughout 2025
   - Confidence intervals for price predictions
   - Identification of potential price floors
4. **Configuration Recommendations**: 
   - Minimum viable specifications for the M4 Max model
   - Cost-benefit analysis of various configuration options
5. **Strategic Recommendations**: 
   - Actionable purchasing strategy with specific timing windows
   - Risk mitigation approaches for potential price volatility
   - Alternative purchasing channels if applicable

## ADDITIONAL PARAMETERS
- All monetary values must be presented in Euro (€)
- All analysis must be specific to the German market, with global context where relevant
- All content must be presented in English
- Citations must be provided for all data sources

## ANALYSIS CONSTRAINTS
- Prioritize accuracy and reliability of information over comprehensiveness
- Acknowledge limitations in predictive capabilities where appropriate
- Maintain objectivity in all assessments and recommendations"""

    if not dry_run:
        response = research_swarm.invoke(
            {"messages": [{"role": "user", "content": content}]},
            {
                "configurable": {
                    "thread_id": "thread-1",
                    "recursion_limit": 1000,
                }
            },
            debug=VERBOSE,
        )

        for m in response["messages"]:
            m.pretty_print()
    else:
        pp("🗺 Draw graph")
        save_lang_graph(
            research_swarm.get_graph(),
            "workflow/lc_BI_swarm(plan_research_swarm).png",
        )

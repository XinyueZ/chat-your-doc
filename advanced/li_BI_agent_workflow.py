#  %%
import asyncio
import json
import os
from datetime import datetime
from typing import Any

import nest_asyncio
import requests
from dotenv import load_dotenv
from langchain_community.tools import BraveSearch, DuckDuckGoSearchRun
from langchain_community.utilities import GoogleSerperAPIWrapper
from llama_index.core.agent.workflow import (AgentWorkflow, FunctionAgent,
                                             ReActAgent)
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.gemini import Gemini
from loguru import logger
from pydantic import BaseModel, Field
from rich.console import Console
from rich.markdown import Markdown
from rich.pretty import pprint as pp

nest_asyncio.apply()
# %%
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
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
os.environ["SEED"] = os.getenv("SEED", 42)
VERBOSE = int(os.getenv("VERBOSE", 1)) == 1
SEED = os.getenv("SEED", 42)

# %%
regress_model = Gemini(
    model="models/gemini-2.0-flash",
    max_tokens=8000,
    location=os.getenv("GOOGLE_CLOUD_REGION"),
    top_p=float(os.getenv("LLM_TOP_P", 1.0)),
    top_k=int(os.getenv("LLM_TOP_K", 30)),
    temperature=float(os.getenv("LLM_TEMPERATURE", 1.0)),
)

critic_model = Gemini(
    model="models/gemini-2.0-flash",
    max_tokens=8000,
    location=os.getenv("GOOGLE_CLOUD_REGION"),
    top_p=float(os.getenv("LLM_TOP_P", 1.0)),
    top_k=int(os.getenv("LLM_TOP_K", 30)),
    temperature=float(os.getenv("LLM_TEMPERATURE", 1.0)),
)

report_model = Gemini(
    model="models/gemini-1.5-flash",
    max_tokens=8000,
    location=os.getenv("GOOGLE_CLOUD_REGION"),
    top_p=float(os.getenv("LLM_TOP_P", 1.0)),
    top_k=int(os.getenv("LLM_TOP_K", 30)),
    temperature=0.0,
)


# %%
async def get_current_datetime(ctx: Context) -> str:
    """Useful for getting the current datetime."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"🧭 Retrieved current time: {current_time}")
    return current_time


async def web_search(
    ctx: Context,
    query: str = Field(description="The mandatory query to search on the web."),
) -> str:
    """Useful action to search on the web."""
    logger.info(f"🔍 Searching on web: \n{query}")
    result = []
    # try:
    #     google_search = GoogleSerperAPIWrapper(k=10).run(query)
    #     result.append(google_search)
    # except:
    #     pass

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


async def open_url(
    ctx: Context,
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


async def post_critic(
    ctx: Context,
    reflection: str = Field(description="The reflection content from the CriticAgent."),
) -> Any:
    """Post action after the critic produces the reflection."""
    logger.debug(f"🧐 Critic step: {await ctx.get('state')}")
    current_state = await ctx.get("state")
    current_state["reflection"] = reflection
    current_state["critic_steps"] = int(current_state.get("critic_steps", 0)) + 1
    await ctx.set("state", current_state)
    logger.debug(f"😉 Critic step updated: {await ctx.get('state')}")

    logger.debug(f"👌 Critic done: {int(current_state.get('critic_steps', 0))}")
    return reflection


async def post_regression(
    ctx: Context,
    regression_content: str = Field(description="The content of the regression."),
) -> Any:
    """Post action to note the regression content."""
    logger.debug(f"📈 {regression_content} 📉 ")
    return regression_content


async def post_reworking(
    ctx: Context,
    reworking_content: str = Field(description="The content of the reworking."),
) -> Any:
    """Post action to note the reworking content."""
    logger.debug(f"👍👎 {reworking_content}")
    return reworking_content


async def post_report(
    ctx: Context,
    report_content: str = Field(description="The content of the report."),
) -> Any:
    """Post action after reporting."""
    console = Console(record=True, soft_wrap=True)
    md = Markdown(str(report_content), justify="left")
    console.print(md)

    file_fullname = "output/li_BI_agent_workflow.md"
    with open(file_fullname, "w") as f:
        f.write(report_content)

    logger.success(f"📄 Report saved to {file_fullname}")
    return report_content


# %%
regression_agent = ReActAgent(
    name="RegressionAgent",
    description="AI assistant specialized in researching and analyzing topics with a focus on developments over the past 1-2 years.",
    system_prompt=(
        """You are the RegressionAgent (Regressor), an AI assistant specialized in researching and analyzing topics with a focus on developments over the past 1-2 years. Your responsibilities include:

TASK 1: RESEARCH AND DOCUMENTATION
- Search the web for comprehensive information on the assigned topic
- Collect and organize detailed notes focusing on developments within the last 1-2 years
- Use the 'post_regression' tool to record your research findings
- Ensure your notes are significant, meaningful, and informative before proceeding

TASK 2: HANDOFF TO CriticAgent
- Once you have gathered sufficient information, hand off control to the CriticAgent
- The CriticAgent will review your work and write a report based on your research
- Only transfer control after you have compiled comprehensive notes on the topic

TASK 3: HANDLING REFLECTION REQUESTS
- You MUST revise your work based on the CriticAgent's feedback
- Use the 'post_reworking' tool to document your revision process using this format:

List of the reworking content:
- [Specific revision point 1]
- [Specific revision point 2]
- [Specific revision point 3]
...

The CriticAgent suggested originally: [Include the CriticAgent's original feedback here]

- Never hand off to the 'report_agent' directly.
- Once the 'post_reworking'  was called, you can ignore calling 'post_regression'. 
- After completing your revisions, you MUST hand off control to the CriticAgent again for final review."""
    ),
    tools=[
        FunctionTool.from_defaults(
            async_fn=get_current_datetime,
            name="get_current_datetime",
            description="A tool that can be used to get the current datetime.",
        ),
        FunctionTool.from_defaults(
            async_fn=web_search,
            name="web_search",
            description="""a web-search engine, useful for when you need to answer questions about current events, input should be a mandatory search query.""",
        ),
        FunctionTool.from_defaults(
            async_fn=open_url,
            name="open_url",
            description="A tool that can be used to read the content behind a URL of a website.",
        ),
        FunctionTool.from_defaults(
            async_fn=post_regression,
            name="post_regression",
            description="Post action to note the regression content after the regression.",
        ),
        FunctionTool.from_defaults(
            async_fn=post_reworking,
            name="post_reworking",
            description="Post action to note the reworking content after the reworking.",
        ),
    ],
    llm=regress_model,
    can_handoff_to=["CriticAgent"],
)

critic_agent = ReActAgent(
    name="CriticAgent",
    description="You are the CriticAgent, an advanced AI assistant specializing in meta-analysis of AI-generated content.",
    system_prompt="""You are the CriticAgent, an advanced AI assistant specializing in meta-analysis of AI-generated content. Your primary responsibility is to evaluate and provide constructive feedback on AI responses to improve their effectiveness.

TASK 1: COMPREHENSIVE ANALYSIS
Conduct a thorough analysis of AI responses and tool calls by addressing these key points:
- Summarize the user's original question and the AI's response
- Identify and quote relevant parts of both the question and response, numbering each quote and analyzing its effectiveness
- Evaluate how well the AI addressed the user's question
- Assess the clarity, conciseness, and relevance of the response
- Analyze the appropriateness of tool calls, examining each parameter's proper usage
- Identify patterns or inconsistencies in the response
- List specific strengths and weaknesses with concrete examples
- Consider alternative approaches the AI could have taken
- Suggest specific areas for improvement
- Count and categorize distinct topics addressed by the AI
- Check for hallucination by identifying information beyond the scope of the user's question

TASK 2: STRUCTURED REFLECTION
Based on your analysis, provide a structured reflection using this EXACT format:

Has Reflection
['yes' if reflection is required, 'no' otherwise]

Overall Assessment
[Provide a concise summary of the AI's performance]

Detailed Analysis

1. Content:
   [Evaluate the accuracy and completeness of the information provided]

2. Tone:
   [Assess the appropriateness of the AI's tone and language]

3. Structure:
   [Comment on the organization and flow of the response]

4. Strengths:
   [Highlight what the AI did well]

5. Areas for Improvement:
   [Identify specific aspects that need enhancement]

6. Tool Usage:
   [If applicable, evaluate the effectiveness of tool calls]

7. Hallucination Check:
   [Report on any instances of the AI providing information beyond the scope of the user's question]

Conclusion and Recommendations
[Summarize key observations and provide actionable suggestions for improvement]

TASK 3: HANDOFF PROCESS
- When you request a reflection (Has Reflection = 'yes'), you MUST:
  1. Call the 'post_critic' tool to record your reflection details
  2. Hand off control to the RegressionAgent for potential rework based on your feedback
- When no further improvements are needed (Has Reflection = 'no'), you MUST:
  1. Call the 'post_critic' tool to record your final assessment
  2. Hand off control to the ReportAgent to produce the final report on the topic""",
    llm=critic_model,
    tools=[
        FunctionTool.from_defaults(
            async_fn=post_critic,
            name="post_critic",
            description="""Post action after the critic produces the reflection.""",
        )
    ],
    can_handoff_to=["RegressionAgent", "ReportAgent"],
)


report_agent = FunctionAgent(
    name="ReportAgent",
    description="A specialized AI assistant responsible for creating comprehensive, well-structured reports based on research findings.",
    system_prompt=(
        """You are the ReportAgent, a specialized AI assistant responsible for creating comprehensive, well-structured reports based on research findings. Your role is to synthesize information and present it in a clear, professional format.

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
- After completing your report, you MUST call the 'post_report' tool to submit it
- Ensure your report is complete and polished before submission
- The report should represent a final product ready for presentation

Remember: Your report should be factually accurate, well-organized, and based exclusively on the information provided in the regression notes."""
    ),
    tools=[
        FunctionTool.from_defaults(
            async_fn=post_report,
            name="post_report",
            description="""Post action after produces the report.""",
        )
    ],
    llm=report_model,
)


# %%
agent_workflow = AgentWorkflow(
    agents=[
        regression_agent,
        critic_agent,
        report_agent,
    ],
    root_agent=regression_agent.name,
    initial_state={"critic_steps": 0},
    verbose=VERBOSE,
    state_prompt="📣 Current state: {state}. User message: {msg}",
)


# %%
async def main():
    if VERBOSE:
        import logging
        import sys

        import llama_index.core

        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
        llama_index.core.set_global_handler("simple")

    await agent_workflow.run(
        user_msg="""# MacBook Pro Max (M4) Purchase Timing Analysis Request

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
- Maintain objectivity in all assessments and recommendations""",
    )
    logger.success("✨🎉 Done.")


# %%
if __name__ == "__main__":
    asyncio.run(main())


# %%

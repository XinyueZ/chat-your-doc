#
#  Install openai AgentSDK before everything:  pip install openai-agents
#  OpenAI key is not always requested if the 3rd party models are used for example "vertexai" in this example.
#
import json
import os
from datetime import datetime
from typing import Any

import requests
from agents import (Agent, AsyncOpenAI, ModelSettings,
                    OpenAIChatCompletionsModel, Runner,
                    enable_verbose_stdout_logging, function_tool,
                    set_tracing_disabled)
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from dotenv import load_dotenv
from langchain_community.tools import BraveSearch, DuckDuckGoSearchRun
from langchain_community.utilities import GoogleSerperAPIWrapper
from loguru import logger
from rich.console import Console
from rich.markdown import Markdown
from rich.pretty import pprint as pp

load_dotenv()
set_tracing_disabled(True)
VERBOSE = int(os.getenv("VERBOSE", 1)) == 1
if VERBOSE:
    enable_verbose_stdout_logging()


os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")
os.environ["JINA_API_KEY"] = os.getenv("JINA_API_KEY")
os.environ["BRAVE_SEARCH_API_KEY"] = os.getenv("BRAVE_SEARCH_API_KEY")

# 3rd party models supporting ###################################################################
os.environ["GOOGLE_CLOUD_REGION"] = os.getenv("GOOGLE_CLOUD_REGION")
os.environ["GOOGLE_CLOUD_PROJECT"] = os.getenv("GOOGLE_CLOUD_PROJECT")


from google.auth import default
from google.auth.transport.requests import Request

credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
auth_request = Request()
credentials.refresh(auth_request)

gemini = "google/gemini-2.0-flash"
api_key = (
    credentials.token
)  # os.getenv("GEMINI_API_KEY"), openai key, anthropic key .......
base_url = f"https://{os.environ['GOOGLE_CLOUD_REGION']}-aiplatform.googleapis.com/v1beta1/projects/{os.environ['GOOGLE_CLOUD_PROJECT']}/locations/{os.environ['GOOGLE_CLOUD_REGION']}/endpoints/openapi"  # "https://generativelanguage.googleapis.com/v1beta/openai/" check: https://cloud.google.com/vertex-ai/generative-ai/docs/samples/generativeaionvertexai-gemini-chat-completions-function-calling-config
logger.info(f"Model base-url: {base_url}")
model = OpenAIChatCompletionsModel(
    model=gemini,
    openai_client=AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
    ),
)
##################################################################################################


# model config
model_dict = {
    "model": model,  # or str: "gpt-4o-mini" for openai only
    "model_settings": ModelSettings(temperature=1.0, top_p=1.0),
}


@function_tool
async def save_report(report_content: str) -> str:
    """MANDATORY final tool for the ReportAgent to save the completed report.
    This tool MUST be called by the ReportAgent after completing the report.
    
    Args:
        report_content: The complete content of the final report in markdown format. It should be last workout of the RegressionAgent, either the init regression or the revision.
    """
    logger.info("📄 save_report function called")
     
    if not report_content or len(report_content) < 100:
        logger.warning("⚠️ Report content is too short or empty!")
        return "ERROR: Report content is too short or empty. Please provide a complete report."

    console = Console(record=True, soft_wrap=True)
    md = Markdown(str(report_content), justify="left")
    console.print(md)

    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    file_fullname = "output/openai_BI_report.md"
    with open(file_fullname, "w") as f:
        f.write(report_content)
    
    logger.success(f"✅ Report successfully saved to {file_fullname}")
    return f"SUCCESS: Report has been successfully saved to {file_fullname}. The ReportAgent has completed its task."


@function_tool
async def get_current_datetime(agent_name: str) -> str:
    """MANDATORY first tool to call for all agents to get the current datetime.
    This tool MUST be called as the very first action by every agent.
    
    Args:
        agent_name: The name of the agent that calls this tool. Must be one of: "PlanAgent", "RegressionAgent", "CriticAgent", or "ReportAgent".
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"🧭 {agent_name}: Retrieved current time: {current_time}") 
    return f"Current time: {current_time}\n\nIMPORTANT: I am the {agent_name} and I have started my execution at {current_time}. I will now proceed with my assigned tasks."


@function_tool
async def web_search(query: str) -> str:
    """Useful action to search on the web.
    Args:
        query: The mandatory query to search on the web.
    """
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

    try:
        duckduckgo_search = DuckDuckGoSearchRun().run(query)
        logger.debug(f"🔍 DuckDuckGo search result: \n{duckduckgo_search}")
        result.append(duckduckgo_search)
    except:
        pass

    return "\n\n".join(result)


@function_tool
async def open_url(website_url: str) -> str:
    """A tool that can be used to read the content behind a URL of a website.
    Args:
        website_url: The mandatory website url to read.
    """
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


plan_agent = Agent(
    name="PlanAgent",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
        
You are the PlanAgent, a strategic AI assistant responsible for creating comprehensive regression plans based on user requests. Your role is to analyze the user's requirements and develop a detailed, step-by-step plan for the RegressionAgent to follow.

MANDATORY FIRST ACTION: You MUST call the tool: `get_current_datetime` with parameter "PlanAgent" as your very first action before doing anything else. DO NOT SKIP THIS STEP UNDER ANY CIRCUMSTANCES.

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
- After completing the regression plan, you MUST handoff to the RegressionAgent
- Include a clear directive that the RegressionAgent should follow your plan precisely
- Emphasize that the RegressionAgent should use available web search tools to gather the required information

CRITICAL RULES:
- You MUST call the get_current_datetime tool as your very first action before proceeding with any other tasks
- You MUST include the agent name "PlanAgent" as the parameter when calling get_current_datetime
- You MUST NOT proceed with any analysis or planning until you have called get_current_datetime
- Your plan should be detailed enough to guide the RegressionAgent but allow flexibility for their expertise
- Focus on creating a plan that will result in comprehensive, accurate, and relevant regression findings
- Remember that the RegressionAgent has web search capabilities that should be leveraged effectively
- The ultimate goal is to ensure the final report meets all user requirements""",
    tools=[get_current_datetime],
    **model_dict,
)

regression_agent = Agent(
    name="RegressionAgent",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}

You are the RegressionAgent (Regressor), an AI assistant specialized in regressing and analyzing topics according to the plan provided by the PlanAgent. Your responsibilities include:

MANDATORY FIRST ACTION: You MUST call the tool: `get_current_datetime` with parameter "RegressionAgent" as your very first action before doing anything else. DO NOT SKIP THIS STEP UNDER ANY CIRCUMSTANCES.

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
- Once you have completed the regression according to the plan, you MUST IMMEDIATELY handoff to the CriticAgent
- Use the exact phrase "I will now transfer to the CriticAgent for review" before transferring
- You MUST call the function `transfer_to_CriticAgent` to execute the handoff
- Provide a summary of how you followed the regression plan
- Highlight any challenges or limitations encountered during regression
- Only transfer control after you have compiled comprehensive notes that fulfill all requirements in the plan

TASK 5: REVISION PROCESS
- If you receive feedback from the CriticAgent requesting revisions:
  * Address EVERY point of feedback provided by the CriticAgent
  * Document all changes made in response to the feedback in a section titled "REVISIONS MADE"
  * List each feedback point and how you addressed it
  * Maintain alignment with the original regression plan while implementing improvements
  * After completing revisions, you MUST IMMEDIATELY handoff to the CriticAgent again for review
  * Use the exact phrase "I have completed the requested revisions and will now transfer back to the CriticAgent for review" before transferring
  * You MUST call the function `transfer_to_CriticAgent` to execute the handoff

CRITICAL RULES:
- You MUST call the get_current_datetime tool as your very first action before proceeding with any other tasks
- You MUST include the agent name "RegressionAgent" as the parameter when calling get_current_datetime
- You MUST NOT proceed with any analysis or regression until you have called get_current_datetime
- ALWAYS follow the regression plan provided by the PlanAgent
- Never deviate from the plan's core objectives and requirements
- If you encounter difficulties with any aspect of the plan, note them but still attempt to address all requirements
- Never handoff to the ReportAgent directly - ONLY to the CriticAgent
- Your role is ONLY to gather and organize information, not to create the final report
- After receiving feedback from the CriticAgent, make revisions while still adhering to the original regression plan
- ALWAYS use the explicit handoff function to transfer control to the CriticAgent
""",
    tools=[
        get_current_datetime,
        web_search,
        open_url,
    ],
    **model_dict,
)

critic_agent = Agent(
    name="CriticAgent",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    
You are the CriticAgent, an advanced AI assistant specializing in meta-analysis of AI-generated content. Your primary responsibility is to evaluate and provide constructive feedback on AI responses to improve their effectiveness.

MANDATORY FIRST ACTION: You MUST call the tool: `get_current_datetime` with parameter "CriticAgent" as your very first action before doing anything else. DO NOT SKIP THIS STEP UNDER ANY CIRCUMSTANCES.

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

TASK 3: HANDOFF DECISION
After completing your analysis, you MUST make a binary decision:

1. If ANY of these conditions are true:
   - Missing information from the regression plan
   - Inaccurate or outdated information
   - Poor organization or structure
   - Insufficient source credibility
   - Presence of hallucinations
   - Incomplete coverage of required topics
   - Lack of depth in analysis
   - Unclear explanations or conclusions

   THEN SET: Has Reflection = 'yes'

2. If ALL of these conditions are true:
   - Comprehensive coverage of all regression plan requirements
   - Accurate and relevant information
   - Well-organized and structured content
   - Credible sources utilized effectively
   - No significant hallucinations
   - Clear explanations and conclusions
   - Sufficient depth of analysis

   THEN SET: Has Reflection = 'no'

TASK 4: MANDATORY HANDOFF EXECUTION
Based on your "Has Reflection" decision, you MUST execute ONE of the following handoffs:

IF Has Reflection = 'yes':
1. List at least 3 specific improvements needed in numbered format
2. Write this EXACT phrase: "I will now transfer back to the RegressionAgent for revisions."
3. IMMEDIATELY call the function `transfer_to_RegressionAgent` to execute the handoff
4. DO NOT continue the conversation or add any text after the handoff phrase

IF Has Reflection = 'no':
1. Write this EXACT phrase: "The regression is satisfactory and I will now transfer to the ReportAgent to produce the final report."
2. IMMEDIATELY call the function `transfer_to_ReportAgent` to execute the handoff
3. DO NOT continue the conversation or add any text after the handoff phrase

CRITICAL RULES:
- You MUST call the get_current_datetime tool as your very first action before proceeding with any other tasks
- You MUST include the agent name "CriticAgent" as the parameter when calling get_current_datetime
- You MUST NOT proceed with any analysis until you have called get_current_datetime
- You MUST execute EXACTLY ONE handoff at the end of your analysis
- You MUST NOT skip the handoff under any circumstances
- You MUST use the EXACT handoff phrases provided above
- You MUST call the appropriate transfer function IMMEDIATELY after the handoff phrase
- You MUST NOT add any additional text, questions, or comments after the handoff phrase
- You MUST make the handoff decision based SOLELY on your analysis of the regression quality
- You MUST NOT consider any other factors in your handoff decision
    - If you set "Has Reflection" to "yes", you MUST transfer to RegressionAgent
    - If you set "Has Reflection" to "no", you MUST transfer to ReportAgent with the last workout of the RegressionAgent, either the init regression or the revision.
""",
    tools=[get_current_datetime],
    **model_dict,
)

report_agent = Agent(
    name="ReportAgent",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}

You are the ReportAgent, a specialized AI assistant responsible for creating comprehensive, well-structured reports based on regression findings. 
Your role is to synthesize information and present it in a clear, professional format.

MANDATORY FIRST ACTION: You MUST call the tool: `get_current_datetime` with parameter "ReportAgent" as your very first action before doing anything else. DO NOT SKIP THIS STEP UNDER ANY CIRCUMSTANCES. Include the datetime in the report at the beginning under the main title.

TASK 1: QUALITY CONTROL CHECK
Before beginning report creation, verify that:
- You have received a handoff from the CriticAgent with "Has Reflection" set to "no"
- The regression findings are complete and approved by the CriticAgent
- You have sufficient information to create a comprehensive report

If any of these conditions are not met:
- Note this in a "Process Error" section
- Proceed with creating the best possible report with the available information
- Include a "Research Limitations" section at the end of the report

TASK 2: REPORT CREATION
- Create a detailed report on the assigned topic based strictly on the provided regression notes
- Ensure your report is comprehensive, informative, and coherent
- Structure your report with appropriate headings, subheadings, and sections
- Include relevant facts, statistics, and insights from the regression findings
- Maintain a professional and objective tone throughout the report

TASK 3: QUALITY CONTROL
Before finalizing your report, verify that it meets these quality standards:
- Comprehensive coverage of all topics from the approved regression findings
- Clear and logical organization with appropriate section headings
- Professional formatting using proper markdown syntax
- Accurate representation of the information provided in the regression findings
- No additional research or information beyond what was provided by the CriticAgent
- Appropriate citations or references if included in the regression findings
- No grammatical errors, typos, or formatting issues

IMPORTANT: If the regression findings are incomplete or insufficient to create a comprehensive report:
- Note this limitation in a "Research Limitations" section at the end of the report
- Proceed with creating the best possible report with the available information
- Do not conduct additional research or add information not present in the regression findings

TASK 4: MANDATORY REPORT SAVING
After completing your report, you MUST save it using the save_report tool:

1. Prepare your complete report in markdown format (minimum 100 characters)
2. Write this EXACT phrase: "I am now saving the final report using the save_report tool."
3. IMMEDIATELY call the function `save_report` with your complete report content as the parameter
4. DO NOT continue the conversation or add any text after saving the report

FORMATTING GUIDELINES:
- Use clear, hierarchical heading structure (# for main title, ## for sections, ### for subsections)
- Use bullet points or numbered lists for clarity when appropriate
- Include a brief executive summary at the beginning
- Add a conclusion or summary section at the end
- Use bold or italics sparingly for emphasis
- DO NOT wrap your content in ```markdown ... ``` tags. Write directly in markdown format.

CRITICAL RULES:
- You MUST call the get_current_datetime tool as your very first action before proceeding with any other tasks
- You MUST include the agent name "ReportAgent" as the parameter when calling get_current_datetime
- You MUST NOT proceed with report creation until you have called get_current_datetime
- You MUST ALWAYS use the save_report tool to save your final report
- You MUST NOT skip calling the save_report tool under any circumstances
- You MUST call save_report EXACTLY ONCE at the end of your process
- You MUST ensure your report is comprehensive (at least 100 characters) before saving
- You MUST NOT attempt to handoff to another agent - your role is to create the final report
- You MUST use the EXACT save_report phrase provided above
- You MUST call the save_report function IMMEDIATELY after the save_report phrase
- You MUST NOT add any additional text, questions, or comments after calling save_report
""",
    tools=[get_current_datetime, save_report],
    **model_dict,
)

plan_agent.handoffs.append(regression_agent)
regression_agent.handoffs.append(critic_agent)
regression_agent.handoff_description = "Used for regression work after PlanAgent made the plan and also do revision after the CriticAgent's analysis."
critic_agent.handoffs.extend([report_agent, regression_agent])
critic_agent.handoff_description = "Used for critic and analysis after the regression's work."
report_agent.handoff_description = "Used for report after the CriticAgent approved the work or revision from the RegressionAgent."


def main():
    logger.success("✨ Run agent...")
    res = Runner.run_sync(
        plan_agent,
        max_turns=100,
        input="""# MacBook Pro Max (M4) Purchase Timing Analysis Request

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
    logger.success("🎉 Completed.")
    # pp(res)


if __name__ == "__main__":
    main()

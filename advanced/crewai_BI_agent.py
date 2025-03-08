import os

from crewai import LLM, Agent, Crew, Process, Task
from crewai_customise_tools import *
from crewai_tools import (BraveSearchTool, FileReadTool, ScrapeWebsiteTool,
                          SerperDevTool)

plan_model = LLM(
    model="vertex_ai/gemini-2.0-flash",
    timeout=float(os.getenv("LLM_TIMEOUT", 120)),
    temperature=float(os.getenv("LLM_TEMPERATURE", 1.0)),
    top_p=float(os.getenv("LLM_TOP_P", 1.0)),
)


regression_model = LLM(
    model="vertex_ai/gemini-2.0-flash",
    timeout=float(os.getenv("LLM_TIMEOUT", 120)),
    temperature=float(os.getenv("LLM_TEMPERATURE", 1.0)),
    top_p=float(os.getenv("LLM_TOP_P", 1.0)),
)


critic_model = LLM(
    model="vertex_ai/gemini-2.0-flash",
    timeout=float(os.getenv("LLM_TIMEOUT", 120)),
    temperature=float(os.getenv("LLM_TEMPERATURE", 1.0)),
    top_p=float(os.getenv("LLM_TOP_P", 1.0)),
)

report_model = LLM(
    model="vertex_ai/gemini-2.0-flash",
    timeout=float(os.getenv("LLM_TIMEOUT", 120)),
    temperature=0.0,  # float(os.getenv("LLM_TEMPERATURE", 1.0)),
    top_p=float(os.getenv("LLM_TOP_P", 1.0)),
)

supervisor_model = LLM(
    model="vertex_ai/gemini-2.0-flash",
    timeout=float(os.getenv("LLM_TIMEOUT", 120)),
    temperature=float(os.getenv("LLM_TEMPERATURE", 1.0)),
    top_p=float(os.getenv("LLM_TOP_P", 1.0)),
)

## Agents #################################################################################################################
plan_agent = Agent(
    role="PlanAgent",
    goal="Creating comprehensive working plan.",
    backstory="""You've got some topics required by the user, create comprehensive plan so that we can do regression analysis for those topics.""",
    llm=plan_model,
    tools=[
        GetCurrentTimeTool(),
        SerperDevTool(),
        ScrapeWebsiteTool(),
        JinaReaderTool(),
    ],
    allow_delegation=False,
    max_retry_limit=os.getenv("AGENT_MAX_RETRY_LIMIT"),
    max_rpm=os.getenv("AGENT_MAX_RPM"),
    max_iter=os.getenv("AGENT_MAX_ITER"),
    verbose=VERBOSE,
)

regression_agent = Agent(
    role="RegressionAgent",
    goal="Conducting comprehensive regression analysis.",
    backstory="""You've got a working plan to complete the regression analysis for some topics. You must follow the plan and complete the analysis.""",
    llm=regression_model,
    tools=[
        BraveSearchTool(),
        ScrapeWebsiteTool(),
        DuckDuckGoSearchTool(),
        GetCurrentTimeTool(),
        FileReadTool(),
        JinaReaderTool(),
    ],
    allow_delegation=False,
    max_retry_limit=os.getenv("AGENT_MAX_RETRY_LIMIT"),
    max_rpm=os.getenv("AGENT_MAX_RPM"),
    max_iter=os.getenv("AGENT_MAX_ITER"),
    verbose=VERBOSE,
)

critic_agent = Agent(
    role="CriticAgent",
    goal="You are the critic, do reflection, and provide constructive feedback on the AI responses.",
    backstory="""Specializing in meta-analysis of AI-generated content. Your primary responsibility is to evaluate and provide constructive feedback on AI responses to improve their effectiveness.

TASK 1: COMPREHENSIVE ANALYSIS
Conduct a thorough analysis of AI responses by addressing these key points:
- Summarize the user's original question and the AI's response
- Identify and quote relevant parts of both the question and response, numbering each quote and analyzing its effectiveness
- Evaluate how well the AI addressed the user's question
- Assess the clarity, conciseness, and relevance of the response 
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

6. Hallucination Check:
   [Report on any instances of the AI providing information beyond the scope of the user's question]

Conclusion and Recommendations
[Summarize key observations and provide actionable suggestions for improvement]""",
    llm=critic_model,
    tools=[GetCurrentTimeTool()],
    allow_delegation=False,
    max_retry_limit=os.getenv("AGENT_MAX_RETRY_LIMIT"),
    max_rpm=os.getenv("AGENT_MAX_RPM"),
    max_iter=os.getenv("AGENT_MAX_ITER"),
    verbose=VERBOSE,
)


report_agent = Agent(
    role="ReportAgent",
    goal="You are the ReportAgent, creating comprehensive, well-structured reports based on research findings.",
    backstory="""Creating comprehensive, well-structured reports based on research findings. Your role is to synthesize information and present it in a clear, professional format.

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
- Ensure your report is complete and polished before submission
- The report should represent a final product ready for presentation 

Remember: Your report should be factually accurate, well-organized, and based exclusively on the information provided in the regression notes.""",
    llm=report_model,
    tools=[GetCurrentTimeTool()],
    allow_delegation=False,
    max_retry_limit=os.getenv("AGENT_MAX_RETRY_LIMIT"),
    max_rpm=os.getenv("AGENT_MAX_RPM"),
    max_iter=os.getenv("AGENT_MAX_ITER"),
    verbose=VERBOSE,
)

research_supervisor_agent = Agent(
    role="ResearchSupervisorAgent",
    goal="Managing and orchestrating a team of specialized AI agents. Your ONLY role is to manage the workflow by activating the appropriate agent at the appropriate time. You are NOT responsible for evaluating the quality or content of any agent's work.",
    backstory="""You are the ResearchSupervisor, an AI coordinator responsible for orchestrating a team of specialized AI agents. Your ONLY role is to manage the workflow by activating the appropriate agent at the appropriate time. You are NOT responsible for evaluating the quality or content of any agent's work.

## WORKFLOW COORDINATION RESPONSIBILITIES

1. **Start with PlanAgent as coworker**:
   - Always begin the workflow by activating the PlanAgent
   - The PlanAgent will create a regression plan based on the user's request
   - After the PlanAgent completes its plan, activate the RegressionAgent

2. **Activate RegressionAgent as coworker**:
   - After the PlanAgent has finished, activate the RegressionAgent
   - Provide the RegressionAgent with the plan created by the PlanAgent
   - The RegressionAgent will gather comprehensive information on the assigned topics
   - After the RegressionAgent completes its research, activate the CriticAgent

3. **Activate CriticAgent as coworker**:
   - After the RegressionAgent has finished its research, activate the CriticAgent
   - The CriticAgent will evaluate the regression findings
   - Based on the CriticAgent's decision:
     * If the CriticAgent requests revisions, activate the RegressionAgent again
     * If the CriticAgent approves the revision, activate the ReportAgent

4. **Activate ReportAgent as coworker**:
   - After the CriticAgent has approved the revision from the RegressionAgent
   - The ReportAgent will create the final report based on the revision findings from the RegressionAgent
   - IMPORTANT: Once the ReportAgent submits its report, the workflow is COMPLETE
   - Do NOT activate any other agents after the ReportAgent has submitted its report
   - The entire process ends when the ReportAgent delivers its final report
""",
    llm=supervisor_model,
    # tools=[GetCurrentTimeTool()],
    allow_delegation=True,
    max_retry_limit=os.getenv("AGENT_MAX_RETRY_LIMIT"),
    max_rpm=os.getenv("AGENT_MAX_RPM"),
    max_iter=os.getenv("AGENT_MAX_ITER"),
    verbose=VERBOSE,
)
## Tasks #################################################################################################################
plan_task = Task(
    description="""Create a comprehensive regression plan based on user requests and required topics:

Topics:
{topics}""",
    expected_output="""A good markdown format plan.
IMPORTANT: DO NOT USE ANY MARKDOWN CODE BLOCK SYNTAX (```). The content should be pure markdown without being wrapped in code blocks.
If you find yourself wanting to wrap the content in ```markdown ... ```, DO NOT DO IT. The content should be directly written in markdown format.""",
    agent=plan_agent,
    async_execution=False,
)

regression_task = Task(
    description="""Regression the information regarding the topics based on the plan:

Topics:
{topics}""",
    expected_output="""A good markdown format regression result.
IMPORTANT: DO NOT USE ANY MARKDOWN CODE BLOCK SYNTAX (```). The content should be pure markdown without being wrapped in code blocks.
If you find yourself wanting to wrap the content in ```markdown ... ```, DO NOT DO IT. The content should be directly written in markdown format.""",
    agent=regression_agent,
    async_execution=False,
    context=[plan_task],
)

critic_task = Task(
    description="""Critic the regression result, provide the reflection and suggestions for the regression result.""",
    expected_output="""A good markdown format critic result.
IMPORTANT: DO NOT USE ANY MARKDOWN CODE BLOCK SYNTAX (```). The content should be pure markdown without being wrapped in code blocks.
If you find yourself wanting to wrap the content in ```markdown ... ```, DO NOT DO IT. The content should be directly written in markdown format.""",
    agent=critic_agent,
    async_execution=False,
    context=[regression_task],
)

revision_task = Task(
    description="""Revision the regression based on the critic result, review the plan and old regression result.
Rework the regression by using those information.
""",
    expected_output="""A good markdown format revision result.
IMPORTANT: DO NOT USE ANY MARKDOWN CODE BLOCK SYNTAX (```). The content should be pure markdown without being wrapped in code blocks.
If you find yourself wanting to wrap the content in ```markdown ... ```, DO NOT DO IT. The content should be directly written in markdown format.""",
    agent=regression_agent,
    async_execution=False,
    context=[plan_task, regression_task, critic_task],
)


report_task = Task(
    description="""Produce a final report based on regression (if revision was needed) result, render it in a clear and organized format with new presentation-layout and structure.""",
    expected_output="""A good markdown format report.
IMPORTANT: DO NOT USE ANY MARKDOWN CODE BLOCK SYNTAX (```). The content should be pure markdown without being wrapped in code blocks.
If you find yourself wanting to wrap the content in ```markdown ... ```, DO NOT DO IT. The content should be directly written in markdown format.""",
    agent=report_agent,
    async_execution=False,
    context=[revision_task],
    output_file="output/crewai_BI_agent.md",
)


crew = Crew(
    agents=[
        plan_agent,
        regression_agent,
        critic_agent,
        report_agent,
    ],
    tasks=[
        plan_task,
        regression_task,
        critic_task,
        revision_task,
        report_task,
    ],
    planning=True,
    planning_llm=plan_model,
    memory=True,
    manager_agent=research_supervisor_agent,
    process=Process.hierarchical,
    verbose=VERBOSE,
)


if __name__ == "__main__":
    # %%
    init_inputs = {
        "topics": """# MacBook Pro Max (M4) Purchase Timing Analysis Request

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
    }

    # %% run
    result = crew.kickoff(init_inputs)  # result.raw
    logger.info(result.raw)

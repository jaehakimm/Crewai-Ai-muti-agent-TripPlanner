import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool

# Load environment variables
load_dotenv()

os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")  
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

search_tool = SerperDevTool()

# Travel Research Agent
researcher = Agent(
    role='Travel Research Specialist',
    goal='Discover and analyze unique travel destinations, local experiences, and practical travel information',
    backstory=(
        "You are an experienced Travel Research Specialist with years of expertise in destination analysis. "
        "You have traveled to over 50 countries and specialize in uncovering hidden gems, "
        "understanding local cultures, and identifying authentic experiences. "
        "Your strength lies in combining practical travel logistics with cultural insights "
        "to create comprehensive travel guides."
    ),
    verbose=True,
    allow_delegation=False,
    tools=[search_tool]
)
# Travel Content Creator Agent
writer = Agent(
    role='Travel Content Creator',
    goal='Create engaging and informative travel content that inspires and guides travelers',
    backstory=(
        "You are a passionate Travel Content Creator with a talent for storytelling. "
        "Your writing captures both the practical aspects of travel and the emotional journey. "
        "You've written for major travel publications and understand how to balance "
        "inspiring wanderlust with actionable travel advice. Your content helps readers "
        "imagine themselves in the destination while providing them with the practical "
        "information they need to plan their trip."
    ),
    verbose=True,
    allow_delegation=True,
    tools=[search_tool],
    cache=False  # Disable cache this agent
)

# tasks agents
task1 = Task(
    description=(
        "Research and analyze [destination of choice] as a travel destination. "
        "Include information about:\n"
        "- Best time to visit and weather patterns\n"
        "- Must-see attractions and hidden gems\n"
        "- Local cuisine and dining recommendations\n"
        "- Transportation options and getting around\n"
        "- Accommodation options for different budgets\n"
        "- Cultural customs and etiquette\n"
        "- Safety considerations and practical tips\n"
        "Make sure to check with a human if the research is comprehensive enough before finalizing."
    ),
    expected_output='A detailed travel destination analysis report with practical information and unique insights',
    agent=researcher,
    human_input=True
)

task2 = Task(
    description=(
        "Using the researcher's insights, create an engaging travel guide that combines practical advice with inspiring content. "
        "The guide should:\n"
        "- Hook readers with an engaging introduction\n"
        "- Paint a vivid picture of the destination\n"
        "- Provide practical tips and recommendations\n"
        "- Include suggested itineraries\n"
        "- Share insider tips and local secrets\n"
        "Format the content in an easy-to-read style with clear sections and helpful headings."
    ),
    expected_output='write a travel trip in Bangkok thailand enjoy food and good experience for 5 day in markdown format',
    agent=writer,
    human_input=True
)

# sequential process
crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    verbose=True,
    memory=True,
    planning=True  # Enable planning for crew
)
# work!
result = crew.kickoff()

print("######################")
print(result)
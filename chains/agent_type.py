from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import os

key = open("../../keys/openaikey.txt").read()
os.environ["OPENAI_API_KEY"] = key



llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

country_template = PromptTemplate(
    input_variables=['country'],
    template="What is the capital of {country}?"
)

gpt_tools = load_tools(["wikipedia", "llm_math"], llm=llm)
agent = initialize_agent(tools=gpt_tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

agent.run("What is the GDP of india currently and what will be the projected gdp if gpd grown rate is 7.2% "
          "for upcoming year?")

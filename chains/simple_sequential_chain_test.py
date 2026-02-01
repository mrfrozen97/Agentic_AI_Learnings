from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
import os

key = open("../../keys/openaikey.txt").read()
os.environ["OPENAI_API_KEY"] = key



llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

country_template = PromptTemplate(
    input_variables=['country'],
    template="What is the capital of {country}?"
)
country_chain = LLMChain(llm=llm, prompt=country_template)

food_template = PromptTemplate(
    input_variables=['capital'],
    template="What is the best food at {capital}?"
)
food_chain = LLMChain(llm=llm, prompt=food_template)


chains = SimpleSequentialChain(chains=[country_chain, food_chain])
print(chains.invoke("Mexico"))

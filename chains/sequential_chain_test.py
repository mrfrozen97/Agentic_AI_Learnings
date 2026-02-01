from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
import os

key = open("../../keys/openaikey.txt").read()
os.environ["OPENAI_API_KEY"] = key



llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

country_template = PromptTemplate(
    input_variables=['country'],
    template="What is the capital of {country}?"
)
country_chain = LLMChain(llm=llm, prompt=country_template, output_key='capital')

food_template = PromptTemplate(
    input_variables=['capital'],
    template="Give a lisr of the best food at {capital}?"
)
food_chain = LLMChain(llm=llm, prompt=food_template, output_key='best_food')


chains = SequentialChain(
    chains=[country_chain, food_chain],
    input_variables=['country'],
    output_variables=['best_food', 'capital']
)
print(chains({'country': 'Mexico'}))

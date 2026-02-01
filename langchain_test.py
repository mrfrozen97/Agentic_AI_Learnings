from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os

key = open("../keys/openaikey.txt").read()
os.environ["OPENAI_API_KEY"] = key


# Instantiate the model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Use the model (e.g., with the invoke method)
# response = llm.invoke("What is the capital of France?")
# print(response.content)

template = PromptTemplate(
    input_variables=['country'],
    template="What is the capital of {country}?"
)

temp_obj = template.format_prompt(country="France")
# response = llm.invoke(temp_obj)

chain = LLMChain(llm=llm, prompt=template)

chain.invoke("India")

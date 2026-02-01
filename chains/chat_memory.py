from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
import os

key = open("../../keys/openaikey.txt").read()
os.environ["OPENAI_API_KEY"] = key

memory = ConversationBufferMemory()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

country_template = PromptTemplate(
    input_variables=['country'],
    template="What is the capital of {country}?"
)

# Asking llm to keep track of memory
# This will send all the past chats to openai for request
country_chain = LLMChain(llm=llm, prompt=country_template, memory=memory)

food_template = PromptTemplate(
    input_variables=['capital'],
    template="What is the best food at {capital}?"
)
food_chain = LLMChain(llm=llm, prompt=food_template, memory=memory)


chains = SimpleSequentialChain(chains=[country_chain, food_chain])

context = chains.run('Mexico')


# Conversation chain

convo_chains = ConversationChain(llm=llm)
convo_chains.run("What is capital of Denmark?")
convo_chains.run("What is 2+5?")
convo_chains.run("Where is the country located?")

# Get the buffer memory
print(convo_chains.memory.buffer)


# Conversation chain but limit the past history
limited_memory = ConversationBufferWindowMemory(k=3)     # Remember last 3 convo
food_chain = LLMChain(llm=llm, memory=limited_memory)


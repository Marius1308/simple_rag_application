from langchain import PromptTemplate
from contextChain import contextChain
from langchain.chat_models.openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

chain = contextChain(
    prompt=PromptTemplate.from_template("{question}"),
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    iterations=3,
)

response = chain.run("What is Langchain?")

print(response)

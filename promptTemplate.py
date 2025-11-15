from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.2,
)

prompt = PromptTemplate.from_template(
    "Jelaskan topik berikut dalam 3 poin bullet: {topic}"
)

chain = prompt | llm

result = chain.invoke({"topic": "Seberapa penting LLM untuk DevOps atau Backend Engineer"})
print(result.content)

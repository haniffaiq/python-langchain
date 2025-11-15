from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(
    model="gpt-4.1-mini",  
    temperature=0.2,
)

res = model.invoke("Jelaskan apa itu LangChain dalam 3 kalimat, bahasa Indonesia yang simpel.")
print(res.content)
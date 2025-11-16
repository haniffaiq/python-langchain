from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import re


def load_doc(path):
    return TextLoader(path, encoding="utf-8").load()


def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    return splitter.split_documents(docs)


def naive_retriever(chunks, query):
    query_words = set(re.findall(r"\w+", query.lower()))
    best = None
    best_score = -1

    for c in chunks:
        words = set(re.findall(r"\w+", c.page_content.lower()))
        score = len(query_words & words)

        if score > best_score:
            best_score = score
            best = c.page_content

    return best


def rag_answer(context, query):
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.2)

    prompt = PromptTemplate.from_template(
        """
Gunakan konteks di bawah ini untuk menjawab pertanyaan.
Jangan gunakan pengetahuan di luar konteks.

Konteks:
{context}

Pertanyaan:
{query}

Jawab singkat dan akurat.
"""
    )

    chain = prompt | llm
    return chain.invoke({"context": context, "query": query}).content


if __name__ == "__main__":
    load_dotenv()

    docs = load_doc("sample_noise.txt")
    chunks = split_docs(docs)

    print("Chunks total:", len(chunks))

    query = "bagaimana scalling di Kubernates"
    best_chunk = naive_retriever(chunks, query)

    print("\n=== Retrieved chunk ===")
    print(best_chunk)

    answer = rag_answer(best_chunk, query)

    print("\n=== Final Answer ===")
    print(answer)

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field, ValidationError
from langchain_core.output_parsers import JsonOutputParser


# ====================================
# STEP 1 — Load & Split
# ====================================

class HPADiagnosis(BaseModel):
    primary_cause: str
    contributing_factors: list[str]
    hpa_misconfig: bool
    suggested_metrics: list[str]

def load_and_split(path):
    loader = TextLoader(path, encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(docs)
    return chunks


# ====================================
# STEP 2 — Build Embeddings
# ====================================

def create_vectorstore(chunks, db_type="faiss"):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"   # murah, efisien
    )

    if db_type == "faiss":
        return FAISS.from_documents(chunks, embeddings)

    if db_type == "chroma":
        return Chroma.from_documents(chunks, embeddings)

    raise ValueError("Invalid db_type")


# ====================================
# STEP 3 — Retrieval
# ====================================

def retrieve(vectorstore, query, k=3):
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k}
    )
    return retriever.get_relevant_documents(query)


# ====================================
# STEP 4 — Ask LLM with context
# ====================================

def ask_llm(context, query):
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.2)
    parser = JsonOutputParser(pydantic_object=HPADiagnosis)

    prompt = PromptTemplate.from_template(
        template="""
Gunakan konteks berikut untuk menjawab pertanyaan.
Jawab HANYA berdasarkan konteks.

Konteks:
{context}

Pertanyaan:
{query}

Analisa issue berikut dan jawab dengan format JSON VALID sesuai schema.\n\n
{format_instructions}\n\n""",
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser
    return chain.invoke({"context": context, "query": query})


# ====================================
# MAIN
# ====================================

if __name__ == "__main__":
    load_dotenv()

    chunks = load_and_split("sample_noise.txt")

    # Vectorstore berbasis semantic
    vs = create_vectorstore(chunks, db_type="faiss")

    query = "apa penyebab HPA tidak melakukan scaling"

    top_chunks = retrieve(vs, query, k=3)

    print("=== RELEVANT CHUNKS ===")
    for i, c in enumerate(top_chunks):
        print(f"\nChunk {i+1}:")
        print(c.page_content)

    merged_context = "\n\n".join(c.page_content for c in top_chunks)

    print("\n=== FINAL ANSWER ===")
    answer = ask_llm(merged_context, query)
    print(answer)

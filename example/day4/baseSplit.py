from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# 1. LOAD DOCUMENT
def load_text(path):
    loader = TextLoader(path, encoding="utf-8")
    return loader.load()


def load_pdf(path):
    loader = PyPDFLoader(path)
    return loader.load()


# 2. SPLIT DOCUMENTS
def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len
    )
    return splitter.split_documents(docs)


if __name__ == "__main__":
    load_dotenv()

    docs = load_text("sample.txt")
    chunks = split_docs(docs)

    print("Total chunks:", len(chunks))
    print("--- Sample chunk ---")
    print(chunks[0].page_content)

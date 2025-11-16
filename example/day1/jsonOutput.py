from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field


# 1. Schema Pydantic
class Explanation(BaseModel):
    topic: str = Field(description="Nama topik yang dijelaskan")
    summary: list[str] = Field(description="Ringkasan dalam list bullet")
    difficulty: str = Field(description="beginner/intermediate/advanced")


def main():
    load_dotenv()

    # 2. Parser berbasis schema
    parser = JsonOutputParser(pydantic_object=Explanation)

    # 3. Model
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.2,
    )

    # 4. PromptTemplate dengan format_instructions
    prompt = PromptTemplate(
        template=(
            "Kamu adalah asisten AI teknis.\n"
            "Jelaskan topik berikut dalam format JSON sesuai schema.\n\n"
            "{format_instructions}\n\n"
            "Topik: {topic}\n"
        ),
        input_variables=["topic"],
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
        },
    )

    # 5. Rangkaian: prompt -> llm -> parser
    chain = prompt | llm | parser

    print("Memanggil chain...\n")

    result = chain.invoke({"topic": "LangChain"})

    print("=== HASIL ===")
    print(result)
    print("\nTipe objek:", type(result))


if __name__ == "__main__":
    main()

from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


# ========== STRICT NESTED SCHEMA ==========
class Analysis(BaseModel):
    cpu: str
    memory: str
    suspicion_level: Literal["low", "medium", "high"]


class DevOpsReport(BaseModel):
    issue: str
    root_cause: str
    impact: str
    risk_level: Literal["low", "medium", "high"]
    analysis: Analysis
    suggestions: list[str]
    estimated_fix_time_minutes: int


# ===========================================
def build_chain():
    parser = JsonOutputParser(pydantic_object=DevOpsReport)

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.1,
    )

    prompt = PromptTemplate(
        template=(
            "Kamu adalah DevOps senior.\n"
            "Analisa issue berikut dan jawab dengan format JSON VALID sesuai schema.\n\n"
            "{format_instructions}\n\n"
            "Issue: {issue}\n"
        ),
        input_variables=["issue"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    return prompt | llm | parser


def troubleshoot(chain, issue: str):
    print(f"\n=== ANALISIS: {issue} ===")

    try:
        result = chain.invoke({"issue": issue})
        print("✅ Output VALID")
        print(result)
        return result
    except ValidationError as ve:
        print("❌ Output TIDAK valid schema")
        print(ve)
        return None


if __name__ == "__main__":
    load_dotenv()
    chain = build_chain()

    tests = [
        "API latency meningkat drastis setelah deploy baru",
        "Port already used ketika deploy Docker Container"
    ]

    for t in tests:
        troubleshoot(chain, t)

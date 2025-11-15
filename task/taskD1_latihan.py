from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()


class DevOpsTroubleshoot(BaseModel):
    issue: str = Field(description="Deskripsi singkat masalah yang terjadi")
    root_cause: str = Field(description="Analisis penyebab utama masalah")
    possible_fixes: list[str] = Field(description="Langkah-langkah perbaikan yang bisa dilakukan")
    risk_level: Literal["low", "medium", "high"] = Field(description="Tingkat risiko jika tidak segera ditangani")
    estimated_fix_time_minutes: int = Field(description="Perkiraan waktu perbaikan dalam menit")


def build_chain():
    parser = JsonOutputParser(pydantic_object=DevOpsTroubleshoot)

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.2,
    )

    prompt = PromptTemplate(
        template=(
            "Kamu adalah DevOps engineer senior.\n"
            "Analisa issue berikut dan jawab DALAM FORMAT JSON sesuai schema.\n\n"
            "{format_instructions}\n\n"
            "Issue: {issue}\n"
        ),
        input_variables=["issue"],
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
        },
    )

    chain = prompt | llm | parser
    return chain


def troubleshoot_issue(chain, issue: str):
    print(f"\n=== ANALISA ISSUE: {issue} ===")
    try:
        result = chain.invoke({"issue": issue})
        print("✅ Parsed successfully.")
        return result
    except ValidationError as ve:
        print("❌ Validation error (schema tidak terpenuhi):")
        print(ve)
    except Exception as e:
        print("❌ Error lain saat memproses issue:")
        print(repr(e))
    return None


def main():
    chain = build_chain()

    issues = [
        "Docker compose service terus restart dengan exit code 137",
        "Pod Kubernetes CrashLoopBackOff karena ImagePullBackOff",
        "Latency API meningkat drastis setelah deploy versi baru",
    ]

    for iss in issues:
        res = troubleshoot_issue(chain, iss)
        if res is not None:
            print(res)
        print("-" * 60)


if __name__ == "__main__":
    main()

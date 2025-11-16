from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


# ==== STRICT SCHEMA ====
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


# ==== BUILD CHAIN ====
def build_chain():
    parser = JsonOutputParser(pydantic_object=DevOpsReport)

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.1,
    )

    prompt = PromptTemplate(
        template=(
            "FORMAT JSON WAJIB VALID.\n"
            "{format_instructions}\n\n"
            "Issue: {issue}"
        ),
        input_variables=["issue"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    return prompt, llm, parser


# ==== SELF HEALING PIPELINE ====
def safe_invoke(prompt, llm, parser, issue):
    try:
        return (prompt | llm | parser).invoke({"issue": issue})
    except ValidationError:
        # try healing
        raw = (prompt | llm).invoke({"issue": issue}).content
        fix_prompt = f"""
Perbaiki JSON berikut agar VALID sesuai schema:

{parser.get_format_instructions()}

JSON rusak:
{raw}
"""
        healed = llm.invoke(fix_prompt).content
        return parser.parse(healed)


if __name__ == "__main__":
    load_dotenv()
    prompt, llm, parser = build_chain()

    result = safe_invoke(prompt, llm, parser, "Server pod mati karena memory leak")
    print(result)

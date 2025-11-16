from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import Literal


# ========== FINAL OUTPUT SCHEMA ==========
class FinalDevOpsReport(BaseModel):
    issue_type: str
    severity: Literal["low", "medium", "high", "critical"]
    root_cause: str
    recommended_actions: list[str]
    estimated_fix_time_minutes: int


def step1_classifier(llm, issue):
    prompt = PromptTemplate.from_template(
        """
Klasifikasikan issue berikut secara singkat.
Berikan dalam format:
- issue_type
- severity (low/medium/high/critical)

Issue:
{issue}

Jawab dengan format:
issue_type: ...
severity: ...
"""
    )
    chain = prompt | llm 
    res = chain.invoke(prompt.invoke({"issue": issue}).text)
    return res.content


def step2_root_cause(llm, issue, classification):
    prompt = PromptTemplate.from_template(
        """
Issue:
{issue}

Classification:
{classification}

Jelaskan akar masalah (root cause) secara teknis dan jelas.
Jawab 3-5 kalimat.
"""
    )
    chain = prompt | llm 
    res = chain.invoke({"issue": issue, "classification": classification})
    return res.content


def step3_action_plan(llm, root_cause):
    prompt = PromptTemplate.from_template(
        """
Root cause:
{root_cause}

Buat rencana tindakan untuk memperbaiki masalah.
Output 3-5 rekomendasi dalam bullet list (tanpa numbering).
"""
    )
    chain = prompt | llm 
    res = chain.invoke(prompt.invoke({"root_cause": root_cause}).text)
    return res.content


def step4_format_json(llm, classification, root_cause, actions):
    class ParserOut(BaseModel):
        issue_type: str
        severity: Literal["low", "medium", "high", "critical"]
        root_cause: str
        recommended_actions: list[str]
        estimated_fix_time_minutes: int

    parser = JsonOutputParser(pydantic_object=ParserOut)

    prompt = PromptTemplate(
        template=(
            "Format seluruh data menjadi JSON valid.\n"
            "{format_instructions}\n\n"
            "Classification:\n{classification}\n\n"
            "Root cause:\n{root_cause}\n\n"
            "Actions:\n{actions}\n"
        ),
        input_variables=["classification", "root_cause", "actions"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm | parser
    res = chain.invoke({
        "classification": classification,
        "root_cause": root_cause,
        "actions": actions,
    })

    return res


def run_pipeline(issue: str):
    load_dotenv()
    llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.1)

    print("=== STEP 1: CLASSIFICATION ===")
    step1 = step1_classifier(llm, issue)
    print(step1)

    print("\n=== STEP 2: ROOT CAUSE ANALYSIS ===")
    step2 = step2_root_cause(llm, issue, step1)
    print(step2)

    print("\n=== STEP 3: ACTION PLAN ===")
    step3 = step3_action_plan(llm, step2)
    print(step3)

    print("\n=== STEP 4: FINAL JSON ===")
    final = step4_format_json(llm, step1, step2, step3)
    print(final)

    return final


if __name__ == "__main__":
    run_pipeline("Kubernetes HPA tidak scaling meskipun CPU 200%")
    # run_pipeline("API latency naik drastis setelah deploy baru")

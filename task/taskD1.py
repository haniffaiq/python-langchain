from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

class DevOpsTroubleshoot(BaseModel):
    issue: str = Field(description="Deskripsi singkat masalah yang terjadi")
    root_cause: str = Field(description="Analisis penyebab utama masalah")
    possible_fixes: list[str] = Field(description="Langkah-langkah perbaikan yang bisa dilakukan")
    risk_level: Literal["low", "medium", "high"] = Field(description="Tingkat risiko jika tidak segera ditangani")
    estimated_fix_time_minutes: int = Field(description="Perkiraan waktu perbaikan dalam menit")

def jsonOutput(issue):
    
    parser = JsonOutputParser(pydantic_object=DevOpsTroubleshoot)

    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.2)

    prompt = PromptTemplate(
        template=(
            "Kamu adalah DevOps senior.\n"
            "Analisa issue berikut dan berikan output dalam format JSON sesuai schema.\n\n"
            "{format_instructions}\n\n"
            "Issue: {issue}\n"
        ),
        input_variables=["issue"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser

    result = chain.invoke({"issue": issue})
    return result


issues = ["Pod Kubernetes CrashLoopBackOff karena ImagePullBackOff","Latency API meningkat drastis setelah deploy versi baru","Disk usage di server hampir 100% dan aplikasi mulai error"]

for issue in issues:
    result = jsonOutput(issue)
    print(result)
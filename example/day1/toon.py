

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from py_toon_format import encode, decode  # TOON


def main():
    load_dotenv()

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.2,
    )

    study_plan = {
        "topics": [
            {"id": 1, "name": "LangChain basics", "priority": "high"},
            {"id": 2, "name": "Prompt engineering", "priority": "high"},
            {"id": 3, "name": "RAG with vector DB", "priority": "medium"},
        ]
    }

    #Python -> TOON
    toon_input = encode(study_plan)

    #Prompt template
    prompt = PromptTemplate.from_template(
        "You are an AI assistant that ONLY works with TOON data.\n\n"
        "INPUT (TOON):\n"
        "```toon\n"
        "{input_toon}\n"
        "```\n\n"
        "TASK:\n"
        "{task_description}\n\n"
        "REQUIREMENTS:\n"
        "- You MUST respond ONLY with a single TOON code block.\n"
        "- The TOON MUST be syntactically valid and decodable.\n"
        "- Do NOT output any natural language explanation outside the TOON block.\n\n"
        "OUTPUT (TOON):\n"
    )

    #Spesifik task
    task = (
        "Transform the topics table by ADDING two new columns:\n"
        '- "estimasi_durasi_jam" (integer, 2–10)\n'
        '- "difficulty" (one of: beginner, intermediate, advanced)\n\n'
        "The header MUST become: "
        '"topics[N]{id,name,priority,estimasi_durasi_jam,difficulty}".\n'
        "For every row, append the two new values in the correct order."
    )

    chain = prompt | llm

    response = chain.invoke({
        "input_toon": toon_input,
        "task_description": task,
    })

    toon_output = response.content.strip()
    print("=== TOON OUTPUT FROM MODEL ===")
    print(toon_output)
    print()

    #TOON -> Python
    decoded = decode(toon_output)
    print("=== DECODED PYTHON OBJECT ===")
    from pprint import pprint
    pprint(decoded)


if __name__ == "__main__":
    main()

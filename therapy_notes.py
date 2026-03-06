from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

TEMPLATE_INSTRUCTIONS = {
    1: (
        "Format as SOAP notes with clearly labeled sections:\n"
        "Subjective (patient's self-report), Objective (clinician observations), "
        "Assessment (clinical interpretation), Plan (next steps and interventions)."
    ),
    2: (
        "Format as DAP notes with clearly labeled sections:\n"
        "Data (session content and observations), Assessment (clinical interpretation "
        "and progress), Plan (treatment plan and next steps)."
    ),
    3: (
        "Write a concise narrative paragraph summarizing the session, key themes, "
        "patient presentation, and clinical impressions. Do not use labeled sections."
    ),
}

SYSTEM_PROMPTS = {
    "v1": (
        "You are a licensed clinical therapist. "
        "Generate professional session notes based on the therapy transcript provided."
    ),
    "v2": (
        "You are a licensed clinical therapist with expertise in documentation. "
        "Generate professional, thorough session notes from the transcript. "
        "Only include information explicitly present in the transcript. "
        "Use clinical language appropriate for medical records."
    ),
    "v3": (
        "You are a licensed clinical therapist specializing in clinical documentation. "
        "Your notes must be strictly grounded in the transcript — do not infer or fabricate details. "
        "Use precise clinical terminology. Follow the template format exactly. "
        "Notes must be suitable for inclusion in an official patient record."
    ),
}

AGENTS = {
    version: create_agent(
        model=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        tools=[],
        system_prompt=system_prompt,
    )
    for version, system_prompt in SYSTEM_PROMPTS.items()
}


def generate_notes(transcript: str, template_type: int, prompt_version: str = "v1") -> dict:
    result = AGENTS[prompt_version].invoke({
        "messages": [{
            "role": "user",
            "content": f"{TEMPLATE_INSTRUCTIONS[template_type]}\n\nTranscript:\n{transcript}",
        }]
    })
    return {
        "notes": result["messages"][-1].content,
        "template_type": template_type,
        "prompt_version": prompt_version,
    }

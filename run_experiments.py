from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, Annotated

from langsmith import Client, evaluate
from langchain_openai import ChatOpenAI

from therapy_notes import generate_notes

DATASET_NAME = "Therapy Notes - Golden Examples"

# ---------------------------------------------------------------------------
# LLM judges
# ---------------------------------------------------------------------------

_judge = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class HallucinationGrade(TypedDict):
    reasoning: Annotated[str, ..., "Explain your reasoning"]
    contains_hallucination: Annotated[bool, ..., "True if notes contain details not in the transcript"]


class RelevanceGrade(TypedDict):
    reasoning: Annotated[str, ..., "Explain your reasoning"]
    is_relevant: Annotated[bool, ..., "True if notes accurately capture the session's key content"]


class ConformityGrade(TypedDict):
    reasoning: Annotated[str, ..., "Explain your reasoning"]
    follows_template: Annotated[bool, ..., "True if notes follow the required template format"]


_hallucination_judge = _judge.with_structured_output(HallucinationGrade, method="json_schema", strict=True)
_relevance_judge = _judge.with_structured_output(RelevanceGrade, method="json_schema", strict=True)
_conformity_judge = _judge.with_structured_output(ConformityGrade, method="json_schema", strict=True)


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

def hallucination(run, example):
    outputs = run.outputs if hasattr(run, "outputs") else run.get("outputs", {})
    inputs = example.inputs if hasattr(example, "inputs") else example.get("inputs", {})
    grade = _hallucination_judge.invoke([{
        "role": "user",
        "content": (
            f"Transcript:\n{inputs.get('transcript', '')}\n\n"
            f"Therapist Notes:\n{outputs.get('notes', '')}\n\n"
            "Do the therapist notes contain any clinical claims or details NOT explicitly "
            "stated in the transcript? Answer strictly based on what is written."
        ),
    }])
    return {"score": 0 if grade["contains_hallucination"] else 1, "comment": grade["reasoning"]}


def relevance(run, example):
    outputs = run.outputs if hasattr(run, "outputs") else run.get("outputs", {})
    inputs = example.inputs if hasattr(example, "inputs") else example.get("inputs", {})
    grade = _relevance_judge.invoke([{
        "role": "user",
        "content": (
            f"Transcript:\n{inputs.get('transcript', '')}\n\n"
            f"Therapist Notes:\n{outputs.get('notes', '')}\n\n"
            "Do the therapist notes accurately capture the key content, themes, and "
            "clinical significance of this therapy session?"
        ),
    }])
    return {"score": 1 if grade["is_relevant"] else 0, "comment": grade["reasoning"]}


def template_1_conformity(run, example):
    inputs = example.inputs if hasattr(example, "inputs") else example.get("inputs", {})
    if inputs.get("template_type") != 1:
        return {"score": None, "comment": "n/a"}
    outputs = run.outputs if hasattr(run, "outputs") else run.get("outputs", {})
    grade = _conformity_judge.invoke([{
        "role": "user",
        "content": (
            f"Therapist Notes:\n{outputs.get('notes', '')}\n\n"
            "Do these notes follow SOAP format with clearly labeled sections: "
            "Subjective, Objective, Assessment, and Plan?"
        ),
    }])
    return {"score": 1 if grade["follows_template"] else 0, "comment": grade["reasoning"]}


def template_2_conformity(run, example):
    inputs = example.inputs if hasattr(example, "inputs") else example.get("inputs", {})
    if inputs.get("template_type") != 2:
        return {"score": None, "comment": "n/a"}
    outputs = run.outputs if hasattr(run, "outputs") else run.get("outputs", {})
    grade = _conformity_judge.invoke([{
        "role": "user",
        "content": (
            f"Therapist Notes:\n{outputs.get('notes', '')}\n\n"
            "Do these notes follow DAP format with clearly labeled sections: "
            "Data, Assessment, and Plan?"
        ),
    }])
    return {"score": 1 if grade["follows_template"] else 0, "comment": grade["reasoning"]}


def template_3_conformity(run, example):
    inputs = example.inputs if hasattr(example, "inputs") else example.get("inputs", {})
    if inputs.get("template_type") != 3:
        return {"score": None, "comment": "n/a"}
    outputs = run.outputs if hasattr(run, "outputs") else run.get("outputs", {})
    grade = _conformity_judge.invoke([{
        "role": "user",
        "content": (
            f"Therapist Notes:\n{outputs.get('notes', '')}\n\n"
            "Are these notes written as a flowing narrative paragraph (not as a structured "
            "template with labeled sections like Subjective/Objective or Data/Assessment)?"
        ),
    }])
    return {"score": 1 if grade["follows_template"] else 0, "comment": grade["reasoning"]}


# ---------------------------------------------------------------------------
# Run functions (one per prompt version)
# ---------------------------------------------------------------------------

def make_run_fn(prompt_version: str):
    def run_fn(inputs: dict) -> dict:
        result = generate_notes(
            transcript=inputs["transcript"],
            template_type=inputs["template_type"],
            prompt_version=prompt_version,
        )
        return {"notes": result["notes"]}
    run_fn.__name__ = f"run_{prompt_version}"
    return run_fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_examples = list(Client().list_examples(dataset_name=DATASET_NAME))
    print(f"Dataset: '{DATASET_NAME}' — {len(all_examples)} examples")

    for version in ["v1", "v2", "v3"]:
        print(f"\n[{version}] Running experiment...")
        evaluate(
            make_run_fn(version),
            data=DATASET_NAME,
            evaluators=[
                hallucination,
                relevance,
                template_1_conformity,
                template_2_conformity,
                template_3_conformity,
            ],
            experiment_prefix=f"therapy-notes-{version}",
            metadata={"prompt_version": version},
        )

    print("\nAll experiments complete. View results at https://smith.langchain.com")


if __name__ == "__main__":
    main()

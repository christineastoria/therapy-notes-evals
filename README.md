# Therapy Notes Demo

This demo shows how to use LangSmith to evaluate an LLM pipeline end-to-end. The pipeline takes a therapy session transcript and generates structured clinical notes. We then run automated evals to check note quality across three different prompt versions and show how to run certain evals on a subset of the dataset. 

## What this demo does

**The pipeline (`therapy_notes.py`)**
A single LLM call (OpenAI `gpt-4o-mini`) that takes two inputs — a therapy transcript and a template type — and returns formatted therapist notes. Every call is automatically traced in LangSmith so you can inspect inputs, outputs, and latency.

**The dataset (`upload_dataset.py`)**
6 hand-written "golden examples" uploaded to LangSmith — 2 examples per template type. Each example has a transcript as input and a reference set of ideal notes as the expected output. These are the ground truth we evaluate against.

If you already have a dataset in LangSmith, see the comment at the top of `upload_dataset.py` for how to use it instead.

**The experiments (`run_experiments.py`)**
Runs 3 experiments — one per prompt version — each scoring every example on 5 evaluators:

| Evaluator | Runs on | What it checks |
|---|---|---|
| `hallucination` | all 6 examples | Do the notes contain anything NOT in the transcript? |
| `relevance` | all 6 examples | Do the notes capture the key content of the session? |
| `template_1_conformity` | template-1 examples only | Are SOAP sections (Subjective/Objective/Assessment/Plan) present? |
| `template_2_conformity` | template-2 examples only | Are DAP sections (Data/Assessment/Plan) present? |
| `template_3_conformity` | template-3 examples only | Is it written as a narrative paragraph (no labeled sections)? |

The template conformity evals return `n/a` for examples of the wrong type, so all 5 evaluators run in a single experiment and you can compare results across prompt versions in one view.

Each evaluator uses `gpt-4o-mini` as an LLM judge with structured output (boolean + reasoning), scored as 0 or 1.

---

## Setup

### 1. Install uv

[uv](https://docs.astral.sh/uv/) is a fast Python package manager. Install it with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then restart your terminal (or run `source ~/.zshrc`).

### 2. Clone the repo and install dependencies

```bash
git clone git@github.com:christineastoria/therapy-notes-evals.git
cd therapy-notes-evals
uv sync
```

`uv sync` reads `pyproject.toml` and installs everything into an isolated virtual environment automatically — no need to manually activate it.

### 3. Set up environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

- **OPENAI_API_KEY** — from [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- **LANGSMITH_API_KEY** — from [smith.langchain.com](https://smith.langchain.com) → Settings → API Keys

The other two values (`LANGSMITH_TRACING=true` and `LANGSMITH_PROJECT`) can stay as-is.

---

## Running the demo

### Step 1 — Upload the golden dataset or pull from existing LangSmith golden dataset 

```bash
uv run python upload_dataset.py
```

This creates a dataset called `Therapy Notes - Golden Examples` in your LangSmith workspace with 6 examples. You only need to run this once.

### Step 2 — Run the experiments

```bash
uv run python run_experiments.py
```

This runs 3 experiments (v1, v2, v3) against the dataset. Each experiment calls the pipeline once per example (6 LLM calls) and then runs all 5 evaluators on each result (up to 30 more LLM judge calls per experiment). Expect it to take 3–5 minutes total.

When each experiment starts, the script prints a direct link to view results in LangSmith:

```
View the evaluation results for experiment: 'therapy-notes-v1-...' at:
https://smith.langchain.com/...
```

Open that link to see a table of scores, per-example results, and the LLM judge's reasoning for each score.

---

## Template types

| Type | Format | Use case |
|---|---|---|
| 1 | SOAP (Subjective, Objective, Assessment, Plan) | Standard clinical documentation |
| 2 | DAP (Data, Assessment, Plan) | Streamlined behavioral health notes |
| 3 | Narrative paragraph | Brief summaries, progress notes |

## Prompt versions

| Version | What changes |
|---|---|
| v1 | Basic instruction: generate professional notes |
| v2 | Adds: only use information from the transcript; use clinical language |
| v3 | Adds: strict grounding requirement; notes must be suitable for official records |

The goal of running v1/v2/v3 is to see whether more prescriptive prompts reduce hallucination and improve template conformity, and whether that comes at any cost to relevance.

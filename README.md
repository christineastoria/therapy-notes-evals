# Therapy Notes Demo

Single LLM call (OpenAI) traced with LangSmith that generates therapist notes from a session transcript. Evaluates across three prompt versions for hallucination, relevance, and template format conformity.

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

This creates a virtual environment and installs all dependencies automatically.

### 3. Set up environment variables

Copy the example env file and fill in your API keys:

```bash
cp .env.example .env
```

Then open `.env` and add your keys:

- **OPENAI_API_KEY** — from [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- **LANGSMITH_API_KEY** — from [smith.langchain.com](https://smith.langchain.com) → Settings → API Keys

### 4. Load your environment

```bash
source .env
```

## Usage

**1. Upload the golden dataset** (6 examples, 2 per template type):
```bash
uv run python upload_dataset.py
```

**2. Run all experiments:**
```bash
uv run python run_experiments.py
```

This runs **12 experiments** total:

| Experiment | Prompt | Evaluators | Examples |
|---|---|---|---|
| `therapy-notes-v1` | v1 | hallucination, relevance | all 6 |
| `therapy-notes-v1-template-1` | v1 | template-1 conformity (SOAP) | 2 |
| `therapy-notes-v1-template-2` | v1 | template-2 conformity (DAP) | 2 |
| `therapy-notes-v1-template-3` | v1 | template-3 conformity (narrative) | 2 |
| `therapy-notes-v2` | v2 | hallucination, relevance | all 6 |
| ... | ... | ... | ... |

## Template Types

| Type | Format |
|---|---|
| 1 | SOAP (Subjective, Objective, Assessment, Plan) |
| 2 | DAP (Data, Assessment, Plan) |
| 3 | Brief narrative paragraph |

## Prompt Versions

| Version | Description |
|---|---|
| v1 | Basic clinical instruction |
| v2 | Adds "only include info from transcript" + clinical language guidance |
| v3 | Adds strict grounding requirement + suitability for official records |

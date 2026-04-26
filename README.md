# Story Generation API

A **FastAPI** service that generates **children's educational fiction stories** (targeting ages 8-12) from a "why?" question or educational theme. The pipeline is designed for producing narrated YouTube-style content: it first creates a **story abstract** and a **story prompt**, then expands them into a full **5-minute storytelling script**, with an optional **multi-model quality-check gate** that can auto-regenerate until the story meets a quality threshold.

---

## Table of Contents

1. [What It Does](#what-it-does)
2. [LLM Models Used](#llm-models-used)
3. [Project Structure](#project-structure)
4. [API Endpoints](#api-endpoints)
   - [GET `/api/v1/health`](#get-apiv1health)
   - [POST `/api/v1/abstract/generate`](#post-apiv1abstractgenerate)
   - [POST `/api/v1/story/generate`](#post-apiv1storygenerate)
   - [POST `/api/v1/story/quality-check`](#post-apiv1storyquality-check)
   - [POST `/api/v1/story/generate-with-quality-gate`](#post-apiv1storygenerate-with-quality-gate)
5. [Quality-Check Debate Workflow](#quality-check-debate-workflow)
6. [Setup & Running](#setup--running)
7. [Environment Variables](#environment-variables)
8. [Example Usage](#example-usage)

---

## What It Does

The API follows a **3-stage pipeline** (with an optional 4th quality gate):

| Stage | Endpoint | Purpose |
|-------|----------|---------|
| 1. Abstract | `POST /api/v1/abstract/generate` | Takes a "why?" theme (e.g. `"Why does the moon change shape?"`) and returns one or more **story abstracts** plus a **story_prompt** — an instruction block that dictates tone, style, and educational delivery for the next stage. |
| 2. Story Generation | `POST /api/v1/story/generate` | Takes the `abstract`, `education_topic`, and `story_prompt` from stage 1 and returns a full **800-950 word storytelling script** (~5 minutes read aloud) plus a **title**. |
| 3. Quality Check *(optional)* | `POST /api/v1/story/quality-check` | Runs a **multi-model editorial debate** where 3 separate LLMs independently review the story, rebut each other across rounds, and produce a **consensus score** (0-100) and summary. |
| 4. Quality Gate *(optional)* | `POST /api/v1/story/generate-with-quality-gate` | Combines stages 2 + 3 into one call. Generates a story, quality-checks it, and **auto-regenerates** (up to `max_regenerations` times) until the consensus score meets the `acceptance_score`. |

All generated content is styled after **Korean "Why?" educational comics** — funny, fast-paced, curiosity-driven, with exaggerated reactions and science woven into the narrative rather than delivered as a lecture.

---

## LLM Models Used

### Story Generation Model (default)

| Property | Value |
|----------|-------|
| **Model** | `MiniMaxAI/MiniMax-M2.5:novita` |
| **Provider** | Hugging Face Router (`https://router.huggingface.co/v1`) |
| **Role** | Generates story abstracts, story prompts, full stories, and titles. |
| **Temperature** | Default `0.7` for stories, `0.85` for abstracts (configurable per request) |

### Quality-Check / Debate Models

The quality-check endpoint runs a **multi-model debate**. Three independent models review the story, then a moderator synthesizes their opinions into a consensus.

| Model | Role |
|-------|------|
| `zai-org/GLM-5:novita` | Independent reviewer |
| `Qwen/Qwen3.5-397B-A17B:novita` | Independent reviewer |
| `openai/gpt-oss-120b:groq` | Independent reviewer + **final consensus moderator** |

These models score the story on a 0-100 rubric covering:
- **Narrative flow** (20 pts)
- **Educational integration** (20 pts)
- **Scientific accuracy** (20 pts)
- **Tone & vocabulary** (15 pts)
- **Read-aloud quality** (10 pts)
- **Character agency** (15 pts)

---

## Project Structure

```
story_api/
│
├── main.py                      # FastAPI app entry point. Sets up CORS and mounts /api/v1 router.
│
├── core/
│   └── config.py                # Configuration: HF_ROUTER_BASE_URL, DEFAULT_MODEL, HF_TOKEN loader.
│
├── routes/
│   ├── v1/
│   │   ├── __init__.py          # Assembles api_v1_router (health + abstract + story).
│   │   ├── health.py            # GET /health endpoint.
│   │   ├── abstract.py          # POST /abstract/generate endpoint.
│   │   └── story.py             # POST /story/generate, /story/quality-check, /story/generate-with-quality-gate.
│   └── __init__.py
│
├── schemas/
│   ├── abstract.py              # Pydantic models: AbstractGenerateRequest, AbstractItem, AbstractOnlyItem.
│   └── story.py                 # Pydantic models: StoryRequest, StoryResponse,
│                                #                    StoryQualityCheckRequest, StoryQualityCheckResponse,
│                                #                    StoryGenerateWithQualityRequest, StoryGenerateWithQualityResponse,
│                                #                    ModelQualityReview.
│
├── services/
│   ├── abstract_service.py      # Business logic for abstract + story_prompt generation.
│   └── story_service.py         # Business logic for story generation, quality-check debate (LangGraph),
│                                # and the generate-with-quality-gate loop.
│
├── .env.example                 # Example environment file showing HF_TOKEN format.
└── README.md                    # This file.
```

---

## API Endpoints

Base path: `/api/v1`

Interactive docs (Swagger UI): `http://127.0.0.1:8000/docs`

### `GET /api/v1/health`

Simple health check. Returns the default model name.

**Response:**
```json
{
  "status": "ok",
  "model": "MiniMaxAI/MiniMax-M2.5:novita"
}
```

---

### `POST /api/v1/abstract/generate`

Generates one or more story abstracts and story prompts from a "why?" educational theme.

**Request Body (`AbstractGenerateRequest`):**

| Field | Type | Required | Default | Constraints | Description |
|-------|------|----------|---------|-------------|-------------|
| `theme` | `string` | Yes | — | `min_length=5` | A central "why?" question or topic, e.g. `"Why does the moon change shape?"` |
| `temperature` | `float` | No | `0.85` | `0.0` – `1.5` | Creativity vs. structure. Higher = more imaginative. |
| `max_tokens` | `integer` | No | `600` | `100` – `1500` | Max length of each generated abstract. |
| `count` | `integer` | No | `1` | `1` – `5` | How many unique abstracts + prompts to generate. |

**Example Request:**
```json
{
  "theme": "Why do we have seasons?",
  "temperature": 0.85,
  "max_tokens": 600,
  "count": 2
}
```

**Response (`list[AbstractItem]`):**
```json
[
  {
    "abstract": "In a cozy classroom...",
    "story_prompt": "Write a fast-paced, warm story..."
  },
  {
    "abstract": "Deep in a magical forest...",
    "story_prompt": "Make it spooky but scientifically curious..."
  }
]
```

- Each `abstract` is a 150-200 word story summary designed as a foundation for a 5-minute video.
- Each `story_prompt` is an actionable instruction block (3-5 sentences) that will be fed to the story generator to control tone, style rules, and educational delivery.

---

### `POST /api/v1/story/generate`

Generates a full 800-950 word storytelling script and a title from an abstract.

**Request Body (`StoryRequest`):**

| Field | Type | Required | Default | Constraints | Description |
|-------|------|----------|---------|-------------|-------------|
| `abstract` | `string` | Yes | — | `min_length=20` | The story abstract (from `/abstract/generate` or custom). |
| `education_topic` | `string` | Yes | — | `min_length=2` | The educational topic to explain, e.g. `"seasons"`. |
| `story_prompt` | `string` | Yes | — | `min_length=10` | The instruction block from `/abstract/generate` that controls tone & style. |
| `temperature` | `float` | No | `0.7` | `0.0` – `1.5` | Story creativity. |
| `max_tokens` | `integer` | No | `1400` | `200` – `4000` | Max output length. |

**Example Request:**
```json
{
  "abstract": "A group of curious kids wonder why it gets cold in winter...",
  "education_topic": "seasons",
  "story_prompt": "Make it warm, adventurous, and fast-paced. Use exaggerated reactions and visual metaphors. Explain Earth's tilt through the characters' trial-and-error mistakes.",
  "temperature": 0.7,
  "max_tokens": 1400
}
```

**Response (`StoryResponse`):**
```json
{
  "story": "It was the first day of winter vacation...",
  "title": "The Tilted Planet Mystery"
}
```

- `story` is plain text narration only (no HTML, Markdown, stage directions, or panel descriptions).
- Target length: 800-950 words (~5 minutes when read aloud).
- Tone: Funny, playfully dramatic, curiosity-driven, warm, fast-paced.
- Must include at least 3 exaggerated `"WHY?!"` moments.

---

### `POST /api/v1/story/quality-check`

Runs a **multi-model editorial debate** to evaluate a story's quality.

**Request Body (`StoryQualityCheckRequest`):**

| Field | Type | Required | Default | Constraints | Description |
|-------|------|----------|---------|-------------|-------------|
| `story` | `string` | Yes | — | `min_length=100` | The story text to evaluate. |
| `story_category` | `string` | No | `"Children's educational fiction"` | `min_length=2` | Category label for evaluation context. |
| `rounds` | `integer` | No | `2` | `1` – `3` | Debate rounds. Round 1 = independent reviews. Later rounds = rebuttals. |
| `temperature` | `float` | No | `0.1` | `0.0` – `1.0` | LLM temperature for reviews (kept low for consistency). |
| `max_tokens` | `integer` | No | `1200` | `300` – `4000` | Max tokens per review response. |

**Example Request:**
```json
{
  "story": "It was the first day of winter vacation...",
  "story_category": "Children's educational fiction",
  "rounds": 2,
  "temperature": 0.1,
  "max_tokens": 1200
}
```

**Response (`StoryQualityCheckResponse`):**

```json
{
  "models": [
    "zai-org/GLM-5:novita",
    "Qwen/Qwen3.5-397B-A17B:novita",
    "openai/gpt-oss-120b:groq"
  ],
  "rounds": 2,
  "consensus_model": "openai/gpt-oss-120b:groq",
  "consensus_score": 82,
  "consensus_summary": "The story effectively integrates scientific concepts...",
  "transcript": [
    {
      "model": "zai-org/GLM-5:novita",
      "round_number": 1,
      "final_score": 80,
      "summary": "Strong narrative flow...",
      "strengths": ["Engaging characters", "Clear explanations"],
      "weaknesses": ["Pacing slows in the middle"],
      "suggested_fix": "Add a mini-conflict in the middle to maintain momentum.",
      "raw_response": "{ ... }"
    }
  ]
}
```

- `consensus_score` is an integer from 0-100 (or `null` if the moderator could not synthesize a score).
- `transcript` contains every review turn from every model across all rounds, including strengths, weaknesses, suggested fixes, and the raw model response.

---

### `POST /api/v1/story/generate-with-quality-gate`

**One-shot endpoint** that generates a story, quality-checks it, and **auto-regenerates** if the consensus score is below the acceptance threshold.

**Request Body (`StoryGenerateWithQualityRequest`):**

| Field | Type | Required | Default | Constraints | Description |
|-------|------|----------|---------|-------------|-------------|
| `abstract` | `string` | Yes | — | `min_length=20` | Story abstract. |
| `education_topic` | `string` | Yes | — | `min_length=2` | Educational topic. |
| `story_prompt` | `string` | Yes | — | `min_length=10` | Story prompt instruction block. |
| `generation_temperature` | `float` | No | `0.7` | `0.0` – `1.5` | Temperature for story generation. |
| `generation_max_tokens` | `integer` | No | `1400` | `200` – `4000` | Max tokens for story generation. |
| `story_category` | `string` | No | `"Children's educational fiction"` | `min_length=2` | Category for quality evaluation. |
| `rounds` | `integer` | No | `2` | `1` – `3` | Debate rounds for quality check. |
| `quality_temperature` | `float` | No | `0.1` | `0.0` – `1.0` | Temperature for quality reviews. |
| `quality_max_tokens` | `integer` | No | `1200` | `300` – `4000` | Max tokens for quality reviews. |
| `acceptance_score` | `integer` | No | `75` | `0` – `100` | Minimum consensus score required to accept the story. |
| `max_regenerations` | `integer` | No | `1` | `0` – `5` | Extra generation attempts allowed if score is below threshold. |

**Example Request:**
```json
{
  "abstract": "A group of curious kids wonder why it gets cold in winter...",
  "education_topic": "seasons",
  "story_prompt": "Make it warm, adventurous, and fast-paced...",
  "generation_temperature": 0.7,
  "generation_max_tokens": 1400,
  "acceptance_score": 80,
  "max_regenerations": 2,
  "rounds": 2
}
```

**Response (`StoryGenerateWithQualityResponse`):**

```json
{
  "accepted": true,
  "attempts": 1,
  "required_score": 80,
  "final_score": 84,
  "story_result": {
    "story": "It was the first day of winter vacation...",
    "title": "The Tilted Planet Mystery"
  },
  "quality_result": {
    "models": ["zai-org/GLM-5:novita", "Qwen/Qwen3.5-397B-A17B:novita", "openai/gpt-oss-120b:groq"],
    "rounds": 2,
    "consensus_model": "openai/gpt-oss-120b:groq",
    "consensus_score": 84,
    "consensus_summary": "Excellent educational integration...",
    "transcript": [ ... ]
  }
}
```

- `accepted` = `true` if the final score met or exceeded `acceptance_score`.
- `attempts` = how many generation + quality-check cycles were run (1 + up to `max_regenerations`).
- If all attempts fail to meet the threshold, `accepted` will be `false` and the best attempt is returned.

---

## Quality-Check Debate Workflow

The quality-check system is implemented as a **LangGraph state machine** with two nodes:

1. **Debate Node** — For each round (1 to `rounds`):
   - Each of the 3 quality models reviews the story independently.
   - In round 2+, each model sees the **latest reviews from the other models** and can rebut or refine its position.
   - Every turn produces a `ModelQualityReview` with: `final_score`, `summary`, `strengths`, `weaknesses`, `suggested_fix`.

2. **Consensus Node** — After all debate rounds:
   - The moderator (`openai/gpt-oss-120b:groq`) receives the full transcript.
   - It synthesizes all opinions into a single **`consensus_score`** (0-100) and a **`consensus_summary`** (2-4 sentences).

This design reduces the bias of any single model by forcing cross-examination before a final judgment is rendered.

---

## Setup & Running

### 1. Create virtual environment & install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Required packages (typical):
- `fastapi`
- `uvicorn`
- `pydantic`
- `langchain-openai`
- `langgraph`
- `python-dotenv`

### 2. Configure your Hugging Face token

Copy `.env.example` to `.env` and set your token:

```
HF_TOKEN="your_hf_token_here"
```

Or set it as an environment variable:

```powershell
$env:HF_TOKEN="your_hf_token_here"
```

The token is used to authenticate with the **Hugging Face Router** (`https://router.huggingface.co/v1`), which proxies requests to the underlying models.

### 3. Run the server

```powershell
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://127.0.0.1:8000`.

- Interactive docs (Swagger UI): `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | **Yes** | Hugging Face API token for router authentication. Also accepts `HugginFaceToken` or `HuggingFaceToken`. |

---

## Example Usage

### Two-step workflow (abstract → story)

**Step 1 — Generate abstract:**
```powershell
curl -X POST "http://127.0.0.1:8000/api/v1/abstract/generate" `
  -H "Content-Type: application/json" `
  -d '{"theme":"Why does the moon change shape?","count":1}'
```

**Step 2 — Generate story from abstract:**
```powershell
curl -X POST "http://127.0.0.1:8000/api/v1/story/generate" `
  -H "Content-Type: application/json" `
  -d '{
    "abstract":"<abstract_from_step_1>",
    "education_topic":"moon phases",
    "story_prompt":"<story_prompt_from_step_1>",
    "temperature":0.7,
    "max_tokens":1400
  }'
```

### One-shot workflow (with quality gate)

```powershell
curl -X POST "http://127.0.0.1:8000/api/v1/story/generate-with-quality-gate" `
  -H "Content-Type: application/json" `
  -d '{
    "abstract":"<your_abstract>",
    "education_topic":"moon phases",
    "story_prompt":"<your_story_prompt>",
    "acceptance_score": 80,
    "max_regenerations": 2,
    "rounds": 2
  }'
```

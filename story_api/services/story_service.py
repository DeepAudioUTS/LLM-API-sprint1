import json
import re
from typing import Any, Optional, TypedDict

from fastapi import HTTPException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from story_api.core.config import DEFAULT_MODEL, HF_ROUTER_BASE_URL, read_hf_token
from story_api.schemas.story import (
    ModelQualityReview,
    StoryGenerateWithQualityRequest,
    StoryGenerateWithQualityResponse,
    StoryQualityCheckRequest,
    StoryQualityCheckResponse,
    StoryRequest,
    StoryResponse,
)


QUALITY_MODELS = [
    "zai-org/GLM-5:novita",
    "Qwen/Qwen3.5-397B-A17B:novita",
    "openai/gpt-oss-120b:groq",
]


class DebateState(TypedDict):
    payload: StoryQualityCheckRequest
    transcript: list[ModelQualityReview]
    consensus_score: int | None
    consensus_summary: str


class StoryService:
    def __init__(self) -> None:
        self.hf_token: Optional[str] = None

    def _get_hf_token(self) -> str:
        if self.hf_token is not None:
            return self.hf_token

        hf_token = read_hf_token()
        if not hf_token:
            raise HTTPException(
                status_code=500,
                detail="Hugging Face token not found. Set HF_TOKEN (or HugginFaceToken) in environment or .env.",
            )

        self.hf_token = hf_token
        return hf_token

    @staticmethod
    def _extract_message_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(parts)
        return str(content or "")

    def _invoke_chat(
        self,
        model: str,
        messages: list[SystemMessage | HumanMessage],
        temperature: float,
        max_tokens: int,
    ) -> str:
        try:
            llm = ChatOpenAI(
                model=model,
                base_url=HF_ROUTER_BASE_URL,
                api_key=self._get_hf_token(),
                temperature=temperature,
                max_tokens=max_tokens,
                request_timeout=180,
            )
            result = llm.invoke(messages)
        except Exception as err:
            raise HTTPException(status_code=502, detail=f"LLM request failed: {err}") from err

        return self._extract_message_text(result.content)

    def generate_story(self, payload: StoryRequest) -> StoryResponse:
        generation_template = PromptTemplate.from_template(
            "# ROLE\n"
            "You are a professional children's storyteller writing for an animated YouTube channel. Your task is to write a funny, educational story about {education_topic}, inspired by the energetic and curiosity-driven style of Korean educational comics like \"Why?\".\n\n"
            "# CRITICAL FORMATTING CONSTRAINTS\n"
            "- Output ONLY plain text story narration.\n"
            "- DO NOT include ANY HTML or XML tags (for example: <p>, <div>, <br>, <!DOCTYPE html>, <html>, <body>).\n"
            "- DO NOT include Markdown formatting (for example: headings, bullet lists, code blocks, or backticks).\n"
            "- The response must be raw text only, as if it were read aloud directly.\n"
            "- DO NOT include: Stage directions, panel descriptions, animation cues, formatting instructions, brackets, production notes, or character names with colons (e.g., \"Narrator:\").\n"
            "- Target Length: 800-950 words (approximately 5 minutes when read aloud).\n"
            "- Pacing: Use a natural spoken storytelling rhythm with lively, dynamic sentences. Avoid overly long paragraphs.\n\n"
            "# TARGET AUDIENCE & TONE\n"
            "- Audience: Children aged 8-12 years old who are curious, easily distracted, love dramatic reactions, and constantly ask \"Why?\".\n"
            "- Tone: Funny, playfully dramatic, curious, warm, and fast-paced.\n"
            "- Style Rules:\n"
            "  - Make it feel like an exciting discovery.\n"
            "  - Include exaggerated reactions and funny misunderstandings.\n"
            "  - Naturally integrate science explanations into the dialogue and events.\n"
            "  - Include at least three (3) exaggerated \"WHY?!\" moments.\n"
            "  - Keep humor playful and silly (strictly no sarcasm or dark humor).\n\n"
            "# EDUCATIONAL CONTENT REQUIREMENTS\n"
            "You must clearly and accurately explain:\n"
            "Abstract source material:\n{abstract}\n\n"
            "Instructions for using the abstract:\n{story_prompt}\n\n"
            "*Explanation Guidelines:* Use simple, highly visual comparisons. Avoid complex physics jargon, textbook-style lectures, and horror tones.\n\n"
            "Now, generate the full 5-minute storytelling script following all instructions above."
        )
        generation_prompt = generation_template.format(
            education_topic=payload.education_topic,
            abstract=payload.abstract,
            story_prompt=payload.story_prompt,
        )

        llm = ChatOpenAI(
            model=DEFAULT_MODEL,
            base_url=HF_ROUTER_BASE_URL,
            api_key=self._get_hf_token(),
            temperature=payload.temperature,
            max_tokens=payload.max_tokens,
            request_timeout=180,
        )

        story_text = ""
        for attempt in range(3):
            try:
                result = llm.invoke([
                    SystemMessage(content="You are an expert children's educational fiction writer."),
                    HumanMessage(content=generation_prompt)
                ])
                story_text = self._extract_message_text(result.content).strip()
                if story_text:
                    break
            except Exception as err:
                if attempt == 2:
                    raise HTTPException(status_code=502, detail=f"LLM request fail: {err}") from err

        if not story_text:
            raise HTTPException(status_code=502, detail="Model returned empty content")

        title_template = PromptTemplate.from_template(
            "Based on the following story about {education_topic}, write a short, catchy, and fun title suitable for children.\n\n"
            "Story:\n{story}\n\n"
            "Return ONLY the title text without any quotes or additional formatting."
        )
        title_prompt = title_template.format(
            education_topic=payload.education_topic,
            story=story_text
        )

        title_text = ""
        for attempt in range(3):
            try:
                result = llm.invoke([
                    SystemMessage(content="You are an expert children's educational fiction writer."),
                    HumanMessage(content=title_prompt)
                ])
                title_text = self._extract_message_text(result.content).strip()
                if title_text:
                    break
            except Exception:
                pass

        if not title_text:
            title_text = "My Curious Story"
        else:
            title_text = title_text.strip('"').strip("'").strip()

        return StoryResponse(story=story_text, title=title_text)

    @staticmethod
    def _extract_json_dict(text: str) -> dict[str, Any] | None:
        if not text:
            return None

        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        for candidate in [cleaned]:
            try:
                parsed = json.loads(candidate, strict=False)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = cleaned[start : end + 1]
            try:
                parsed = json.loads(candidate, strict=False)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                return None

        return None

    def _coerce_int(value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(round(float(value)))
        if isinstance(value, str):
            match = re.search(r"-?\d+(?:\.\d+)?", value)
            if match:
                return int(round(float(match.group(0))))
        return None

    def _chat(self, model: str, prompt: str, temperature: float, max_tokens: int) -> str:
        return self._invoke_chat(
            model=model,
            messages=[
                SystemMessage(
                    content=(
                        "You are an expert children's educational fiction editor. "
                        "Return valid JSON only when requested."
                    )
                ),
                HumanMessage(content=prompt),
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _review_turn(
        self,
        model: str,
        payload: StoryQualityCheckRequest,
        round_number: int,
        others_latest: list[ModelQualityReview],
    ) -> ModelQualityReview:
        others_context = "\n".join(
            [
                (
                    f"- {r.model}: score={r.final_score}, summary={r.summary}, "
                    f"weaknesses={'; '.join(r.weaknesses) if r.weaknesses else 'n/a'}"
                )
                for r in others_latest
            ]
        )
        if not others_context:
            others_context = "- No prior reviewer comments."

        review_template = PromptTemplate.from_template(
            "Evaluate story quality for ages 8-12 with strict scoring.\n"
            "Story category: {story_category}\n"
            "Debate round: {round_number}\n"
            "Other reviewers' current positions:\n"
            "{others_context}\n\n"
            "Rubric weights: narrative_flow(20), educational_integration(20), scientific_accuracy(20), tone_vocabulary(15), read_aloud(10), character_agency(15).\n"
            "If round > 1, respond to disagreements and refine your score.\n"
            "Return JSON only with keys: final_score, summary, strengths, weaknesses, suggested_fix.\n\n"
            "Story:\n"
            "{story}"
        )
        prompt = review_template.format(
            story_category=payload.story_category,
            round_number=round_number,
            others_context=others_context,
            story=payload.story,
        )

        raw = self._chat(
            model=model,
            prompt=prompt,
            temperature=payload.temperature,
            max_tokens=payload.max_tokens,
        )
        parsed = self._extract_json_dict(raw) or {}

        summary = str(parsed.get("summary") or "No summary provided.").strip()
        strengths = parsed.get("strengths") if isinstance(parsed.get("strengths"), list) else []
        weaknesses = parsed.get("weaknesses") if isinstance(parsed.get("weaknesses"), list) else []
        suggested_fix = parsed.get("suggested_fix")

        return ModelQualityReview(
            model=model,
            round_number=round_number,
            final_score=self._coerce_int(parsed.get("final_score")),
            summary=summary,
            strengths=[str(x) for x in strengths],
            weaknesses=[str(x) for x in weaknesses],
            suggested_fix=str(suggested_fix) if suggested_fix is not None else None,
            raw_response=raw,
        )

    def quality_check_story(self, payload: StoryQualityCheckRequest) -> StoryQualityCheckResponse:
        try:
            from langgraph.graph import END, START, StateGraph
        except Exception as err:  # pragma: no cover - environment dependency
            raise HTTPException(
                status_code=500,
                detail=f"LangGraph is required for quality-check workflow: {err}",
            ) from err

        def debate_node(state: DebateState) -> DebateState:
            transcript: list[ModelQualityReview] = list(state.get("transcript", []))
            latest_by_model: dict[str, ModelQualityReview] = {}

            for round_number in range(1, payload.rounds + 1):
                for model in QUALITY_MODELS:
                    others_latest = [
                        review
                        for m, review in latest_by_model.items()
                        if m != model
                    ]
                    turn = self._review_turn(
                        model=model,
                        payload=payload,
                        round_number=round_number,
                        others_latest=others_latest,
                    )
                    latest_by_model[model] = turn
                    transcript.append(turn)

            return {
                "payload": payload,
                "transcript": transcript,
                "consensus_score": state.get("consensus_score"),
                "consensus_summary": state.get("consensus_summary", ""),
            }

        def consensus_node(state: DebateState) -> DebateState:
            compact_transcript = [
                {
                    "model": review.model,
                    "round_number": review.round_number,
                    "final_score": review.final_score,
                    "summary": review.summary,
                    "strengths": review.strengths,
                    "weaknesses": review.weaknesses,
                    "suggested_fix": review.suggested_fix,
                }
                for review in state.get("transcript", [])
            ]

            consensus_template = PromptTemplate.from_template(
                "You are the final moderator in a multi-model editorial debate.\n"
                "Synthesize the debate into one consensus judgment for story quality.\n"
                "Return JSON only with keys: consensus_score, consensus_summary.\n"
                "Keep summary to 2-4 sentences.\n\n"
                "Debate transcript JSON:\n{debate_transcript_json}"
            )
            prompt = consensus_template.format(
                debate_transcript_json=json.dumps(compact_transcript, ensure_ascii=True)
            )

            raw = self._chat(
                model="openai/gpt-oss-120b:groq",
                prompt=prompt,
                temperature=0.0,
                max_tokens=500,
            )
            parsed = self._extract_json_dict(raw) or {}

            return {
                "payload": payload,
                "transcript": state.get("transcript", []),
                "consensus_score": self._coerce_int(parsed.get("consensus_score")),
                "consensus_summary": str(parsed.get("consensus_summary") or "Consensus could not be summarized."),
            }

        graph_builder = StateGraph(DebateState)
        graph_builder.add_node("debate", debate_node)
        graph_builder.add_node("consensus", consensus_node)
        graph_builder.add_edge(START, "debate")
        graph_builder.add_edge("debate", "consensus")
        graph_builder.add_edge("consensus", END)

        graph = graph_builder.compile()
        final_state = graph.invoke(
            {
                "payload": payload,
                "transcript": [],
                "consensus_score": None,
                "consensus_summary": "",
            }
        )

        return StoryQualityCheckResponse(
            models=QUALITY_MODELS,
            rounds=payload.rounds,
            consensus_model="openai/gpt-oss-120b:groq",
            consensus_score=final_state.get("consensus_score"),
            consensus_summary=final_state.get("consensus_summary") or "Consensus could not be summarized.",
            transcript=final_state.get("transcript", []),
        )

    def generate_story_with_quality_gate(
        self, payload: StoryGenerateWithQualityRequest
    ) -> StoryGenerateWithQualityResponse:
        attempts = 0
        final_story: StoryResponse | None = None
        final_quality: StoryQualityCheckResponse | None = None

        total_attempts = payload.max_regenerations + 1
        for _ in range(total_attempts):
            attempts += 1

            story_result = self.generate_story(
                StoryRequest(
                    abstract=payload.abstract,
                    education_topic=payload.education_topic,
                    story_prompt=payload.story_prompt,
                    temperature=payload.generation_temperature,
                    max_tokens=payload.generation_max_tokens,
                )
            )

            quality_result = self.quality_check_story(
                StoryQualityCheckRequest(
                    story=story_result.story,
                    story_category=payload.story_category,
                    rounds=payload.rounds,
                    temperature=payload.quality_temperature,
                    max_tokens=payload.quality_max_tokens,
                )
            )

            final_story = story_result
            final_quality = quality_result

            if (
                quality_result.consensus_score is not None
                and quality_result.consensus_score >= payload.acceptance_score
            ):
                return StoryGenerateWithQualityResponse(
                    accepted=True,
                    attempts=attempts,
                    required_score=payload.acceptance_score,
                    final_score=quality_result.consensus_score,
                    story_result=story_result,
                    quality_result=quality_result,
                )

        if final_story is None or final_quality is None:
            raise HTTPException(status_code=500, detail="Story generation workflow produced no result")

        return StoryGenerateWithQualityResponse(
            accepted=False,
            attempts=attempts,
            required_score=payload.acceptance_score,
            final_score=final_quality.consensus_score,
            story_result=final_story,
            quality_result=final_quality,
        )


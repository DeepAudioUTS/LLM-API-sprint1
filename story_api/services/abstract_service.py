import json
import re
from typing import Any

from fastapi import HTTPException
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from story_api.core.config import DEFAULT_MODEL, HF_ROUTER_BASE_URL, read_hf_token
from story_api.schemas.abstract import AbstractGenerateRequest, AbstractItem, AbstractOnlyItem


def _extract_json_dict(text: str) -> dict[str, Any] | None:
    """Extract a JSON object from LLM output, handling code fences and extra text."""
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return None


class AbstractService:
    def __init__(self) -> None:
        self._hf_token: str | None = None

    def _get_hf_token(self) -> str:
        if self._hf_token is not None:
            return self._hf_token
        token = read_hf_token()
        if not token:
            raise HTTPException(
                status_code=500,
                detail="Hugging Face token not found. Set HF_TOKEN (or HugginFaceToken) in environment or .env.",
            )
        self._hf_token = token
        return token

    def _generate_story_prompt_from_abstract(self, abstract: str, theme: str) -> str:
        """Fallback: generate story_prompt when the main call did not return it."""
        prompt = (
            f"Given this story abstract and theme, write a brief instruction (1–3 sentences) "
            f"for transforming the abstract into a full 5-minute children's story (ages 8–12). "
            f"Include tone (e.g. playful, funny), structure, and how to weave in the educational content.\n\n"
            f"Theme: {theme}\n\nAbstract:\n{abstract}\n\n"
            "Return only the story prompt text, nothing else. No JSON, no labels."
        )
        try:
            llm = ChatOpenAI(
                model=DEFAULT_MODEL,
                base_url=HF_ROUTER_BASE_URL,
                api_key=self._get_hf_token(),
                temperature=0.7,
                max_tokens=200,
                request_timeout=60,
            )
            result = llm.invoke([HumanMessage(content=prompt)])
            return self._extract_message_text(result.content).strip()
        except Exception as err:
            raise HTTPException(
                status_code=502, detail=f"Failed to generate story prompt: {err}"
            ) from err

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

    @staticmethod
    def _extract_abstract_text(raw: str, parsed: dict[str, Any] | None) -> str:
        """Extract clean abstract text even when the model returns malformed JSON-like output."""
        if parsed:
            candidate = str(parsed.get("abstract") or "").strip()
            if candidate:
                return candidate

        cleaned = (raw or "").strip()
        if not cleaned:
            return ""

        # Handle cases where the model returns text like: {"abstract":"...", "story_prompt":"..."
        abstract_match = re.search(r'"abstract"\s*:\s*"(?P<value>.*?)"(?:\s*,\s*"story_prompt"|\s*})', cleaned, re.DOTALL)
        if abstract_match:
            candidate = abstract_match.group("value")
            candidate = candidate.replace('\\n', ' ').replace('\\"', '"').strip()
            if candidate:
                return candidate

        return cleaned

    def _generate_single_abstract(
        self, payload: AbstractGenerateRequest
    ) -> AbstractItem:
        """Generate one abstract and story_prompt pair."""
        output_format = (
            "\n\nOUTPUT FORMAT (STRICT): Return valid JSON only with these keys:\n"
            '- "abstract": A short story abstract (2–4 sentences), plain text.\n'
            '- "story_prompt": A prompt instructing how to turn the abstract into a full 5-minute '
            "children's story (ages 8–12). Include tone, structure, and educational goals.\n"
            "Example: {\"abstract\": \"...\", \"story_prompt\": \"...\"}"
        )

        prompt = (
            "You are a professional children's storyteller creating story abstracts.\n\n"
            "Generate BOTH an abstract AND a story prompt based on this theme or question:\n"
            f"**{payload.theme}**\n\n"
            "INSTRUCTIONS FOR ABSTRACT:\n"
            "- Base it entirely on the theme/'why?' above.\n"
            "- Be creative and varied: introduce plausible characters, settings, or angles.\n"
            "- Maintain high story quality: clear premise, age-appropriate language, educational potential.\n"
            "- The abstract should be usable as source material for a full 5-minute children's story (ages 8–12).\n"
            "- Include a hook that makes the reader want to hear the full story.\n\n"
            "INSTRUCTIONS FOR STORY_PROMPT:\n"
            "- Write a brief instruction (1–3 sentences) for transforming the abstract into the full story.\n"
            "- Specify tone (e.g. playful, funny), target age, and how to weave in the educational content.\n"
        ) + output_format

        try:
            llm = ChatOpenAI(
                model=DEFAULT_MODEL,
                base_url=HF_ROUTER_BASE_URL,
                api_key=self._get_hf_token(),
                temperature=payload.temperature,
                max_tokens=payload.max_tokens,
                request_timeout=120,
            )
            result = llm.invoke([HumanMessage(content=prompt)])
            raw = self._extract_message_text(result.content)
        except Exception as err:
            raise HTTPException(status_code=502, detail=f"LLM request failed: {err}") from err

        parsed = _extract_json_dict(raw)
        abstract_text = ""
        story_prompt_text = ""

        abstract_text = self._extract_abstract_text(raw, parsed)
        if parsed:
            story_prompt_text = str(parsed.get("story_prompt") or "").strip()
        if not abstract_text:
            raise HTTPException(status_code=502, detail="Model returned empty content")

        if not story_prompt_text:
            story_prompt_text = self._generate_story_prompt_from_abstract(
                abstract_text, payload.theme
            )

        return AbstractItem(abstract=abstract_text, story_prompt=story_prompt_text)

    def generate_abstract(
        self, payload: AbstractGenerateRequest
    ) -> list[AbstractItem]:
        """
        Generate one or more story abstracts with story prompts from a theme or "why?" question.
        Uses randomness (via temperature) to produce varied abstracts
        while maintaining story quality through the prompt.
        """
        abstracts: list[AbstractItem] = []
        for _ in range(payload.count):
            item = self._generate_single_abstract(payload)
            abstracts.append(item)
        return abstracts

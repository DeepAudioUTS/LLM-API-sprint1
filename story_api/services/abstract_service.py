import json
import re
from typing import Any

from fastapi import HTTPException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from story_api.core.config import DEFAULT_MODEL, HF_ROUTER_BASE_URL, read_hf_token
from story_api.schemas.abstract import AbstractGenerateRequest, AbstractItem, AbstractOnlyItem




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

    def _generate_story_prompt_from_abstract(self, abstract: str, theme: str) -> str:

        prompt = f"""Given this story abstract and theme, write an actionable, precise instruction block (3–5 sentences) that will be fed to a secondary AI writer to generate the full script. This prompt must dictate:
            - Tone & Vibe: Specify the exact emotional feel (e.g., fast-paced and goofy, scientifically curious but spooky, warm and adventurous).
            - Stylistic Rules: Mandate specific writing techniques (e.g., "Use exaggerated reactions," "Include fast-paced dialogue," "Focus on sensory details of the environment").
            - Educational Delivery: Give the writer strict instructions on how to weave the facts in naturally (e.g., "Explain the science through the character's trial-and-error mistakes," "Do not use textbook definitions; use visual metaphors").

            Theme: {theme}

            Abstract:
            {abstract}

            Write the instructions directly without any conversational filler."""
            
        llm = ChatOpenAI(
            model=DEFAULT_MODEL,
            base_url=HF_ROUTER_BASE_URL,
            api_key=self._get_hf_token(),
            temperature=0.7,
            max_tokens=500,
            request_timeout=60,
        )
        for attempt in range(3):
            try:
                messages = [
                    SystemMessage(content="You are an expert AI prompt engineer and story architect."),
                    HumanMessage(content=prompt)
                ]
                result = llm.invoke(messages)
                text = self._extract_message_text(result.content).strip()
                if text:
                    return text
            except Exception as err:
                if attempt == 2:
                    raise HTTPException(status_code=502, detail=f"Failed to generate story prompt: {err}") from err
        
        raise HTTPException(status_code=502, detail="Failed to generate story prompt: Model returned empty content after 3 attempts.")
    
    def _generate_single_abstract(
        self, payload: AbstractGenerateRequest
    ) -> AbstractItem:
        prompt = f"""
            # ROLE & OBJECTIVE
            You are an elite narrative designer and award-winning children's author specializing in educational content. Your task is to act as the "Story Architect" for a YouTube channel targeting middle-grade readers. You will generate a high-concept Story Abstract based on a provided theme, heavily inspired by the energetic style of Korean "Why?" educational comics.

            # THEME, TONE & TARGET AUDIENCE
            - Theme / Core Question: **{payload.theme}**
            - Target Audience: Children aged 8–12. They are naturally curious, love dramatic reactions, and quickly lose interest if a story feels like a textbook lecture.
            - Tone & Vibe: Funny, playfully dramatic, fast-paced, and deeply curious. Capture the essence of Korean "Why?" comics—expect exaggerated reactions, funny misunderstandings, and explosive moments of scientific discovery.

            # INSTRUCTIONS FOR THE STORY ABSTRACT
            You must write a comprehensive story summary (150–200 words) that serves as the foundation for a 5-minute narrated video.
            - The Hook: Begin with an exciting, highly dramatic, or goofy 1-2 sentence hook that sets up an immediate, absurd mystery.
            - The Protagonist: Introduce a clear main character who is highly inquisitive, prone to exaggerated comic-style reactions, and constantly asking "WHY?!".
            - The Setting: Establish a vivid, imaginative, or highly stylized world.
            - The Conflict & Stakes: Clearly define a wildly dramatic (but kid-friendly) problem. What goes wrong, and what funny disaster happens if they fail?
            - Educational Integration: The learning element MUST be the key to solving the conflict. The protagonist must discover the core science of the theme (**{payload.theme}**) through trial-and-error, hilarious misunderstandings, or dramatic "Aha!" revelations. It cannot be a tacked-on lecture.

            Return ONLY the abstract text. Do not use JSON formatting or code blocks.
            """
        llm = ChatOpenAI(
            model=DEFAULT_MODEL,
            base_url=HF_ROUTER_BASE_URL,
            api_key=self._get_hf_token(),
            temperature=payload.temperature,
            max_tokens=payload.max_tokens,
            request_timeout=120,
        )

        abstract_text = ""
        for attempt in range(3):
            try:
                messages = [
                    SystemMessage(content="You are an elite narrative designer and award-winning children's author specializing in educational content."),
                    HumanMessage(content=prompt)
                ]
                result = llm.invoke(messages)
                abstract_text = self._extract_message_text(result.content).strip()
                if abstract_text:
                    break
            except Exception as err:
                if attempt == 2:
                    raise HTTPException(status_code=502, detail=f"LLM request fail: {err}") from err
                    
        if not abstract_text:
            raise HTTPException(status_code=502, detail="Failed to generate abstract: Model returned empty content after 3 attempts.")

        story_prompt_text = self._generate_story_prompt_from_abstract(abstract_text, payload.theme)

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

import asyncio
from langchain_core.messages import HumanMessage
from story_api.services.story_service import StoryService
from story_api.core.config import DEFAULT_MODEL

async def test():
    svc = StoryService()
    
    json_guardrail = (
        "\n\nOUTPUT FORMAT REQUIREMENTS (STRICT):\n"
        "- Return valid JSON only.\n"
        '- Use exactly these keys: "title", "story".\n'
        "- Do not return Markdown, code fences, or any extra keys.\n"
    )

    generation_prompt = (
        "You are a professional children's storyteller.\n\n"
        "Write a fun and educational story for children ages 8-12.\n"
        "Education topic: Photosynthesis\n\n"
        "Abstract source material:\n"
        "A plant named Leafy learns how to make food from the sun.\n\n"
        "Instructions for using the abstract:\n"
        "Write an adventurous story about Leafy the plant. Emphasize sunlight.\n\n"
        "Also generate a short, catchy title suitable for children.\n"
    )

    raw_output = svc._invoke_chat(
        model=DEFAULT_MODEL,
        messages=[HumanMessage(content=generation_prompt + json_guardrail)],
        temperature=0.7,
        max_tokens=1000,
    )
    print("RAW OUTPUT:")
    print(repr(raw_output))
    
    parsed = svc._extract_json_dict(raw_output)
    print("PARSED:")
    print(parsed)

if __name__ == "__main__":
    asyncio.run(test())

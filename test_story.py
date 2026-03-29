import asyncio
from story_api.schemas.story import StoryRequest
from story_api.services.story_service import StoryService

async def test():
    svc = StoryService()
    req = StoryRequest(
        education_topic="Photosynthesis",
        abstract="A plant named Leafy learns how to make food from the sun.",
        story_prompt="Write an adventurous story about Leafy the plant. Emphasize sunlight.",
        temperature=0.7,
        max_tokens=1000
    )
    res = svc.generate_story(req)
    print("FINISHED")
    print("TITLE:", repr(res.title))
    print("STORY (first 100 chars):", repr(res.story[:100]))

if __name__ == "__main__":
    asyncio.run(test())

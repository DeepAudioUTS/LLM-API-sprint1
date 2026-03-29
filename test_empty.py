import asyncio
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from story_api.core.config import DEFAULT_MODEL, HF_ROUTER_BASE_URL, read_hf_token

async def test():
    token = read_hf_token()
    
    llm = ChatOpenAI(
        model=DEFAULT_MODEL,
        base_url=HF_ROUTER_BASE_URL,
        api_key=token,
        temperature=0.7,
        max_tokens=500,
        request_timeout=60,
    )
    
    abstract = "A brave dog explores the moon and discovers cheese."
    theme = "Moon Science"
    
    prompt = f"""Given this story abstract and theme, write an actionable, precise instruction block (3-5 sentences) that will be fed to a secondary AI writer to generate the full script. This prompt must dictate:
        - Tone & Vibe: Specify the exact emotional feel (e.g., fast-paced and goofy, scientifically curious but spooky, warm and adventurous).
        - Stylistic Rules: Mandate specific writing techniques (e.g., "Use exaggerated reactions," "Include fast-paced dialogue," "Focus on sensory details of the environment").
        - Educational Delivery: Give the writer strict instructions on how to weave the facts in naturally (e.g., "Explain the science through the character's trial-and-error mistakes," "Do not use textbook definitions; use visual metaphors").

        Theme: {theme}

        Abstract:
        {abstract}

        Write the instructions directly without any conversational filler."""
        
    try:
        # TEST 1: System Message + Human Message
        print("TEST 1: System Message + Human Message")
        response = llm.invoke([
            SystemMessage(content="You are a helpful AI assistant that writes instructions for other tools."),
            HumanMessage(content=prompt)
        ])
        print(repr(response.content))
        
        # TEST 2: Altered prompt without "secondary AI writer" trigger
        print("\nTEST 2: Altered prompt")
        prompt2 = f"""Write an actionable, precise instruction block (3-5 sentences) for a story based on the theme and abstract below. The instructions must dictate:
        - Tone & Vibe: Specify the exact emotional feel.
        - Stylistic Rules: Mandate specific writing techniques.
        - Educational Delivery: Explain how to weave facts in naturally.

        Theme: {theme}

        Abstract:
        {abstract}

        Return strictly the instructions."""
        response2 = llm.invoke([HumanMessage(content=prompt2)])
        print(repr(response2.content))
        
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(test())

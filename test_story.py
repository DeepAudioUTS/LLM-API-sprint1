import os
from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key="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",
)
def generate_story(prompt, model, max_tokens=2000): # Increased to 2000 tokens!
 
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a creative assistant that writes fun, simple children's stories with educational themes."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.8,
        max_tokens=max_tokens
    )
    
    message = response.choices[0].message
    
    final_output = ""
    
    if hasattr(message, 'reasoning_content') and message.reasoning_content:
        final_output += "🧠 --- MODEL'S THINKING PROCESS ---\n"
        final_output += message.reasoning_content + "\n\n"
        final_output += "📖 --- FINAL STORY ---\n"
        
    if message.content:
        final_output += message.content
    elif not message.content and getattr(message, 'reasoning_content', None):
        final_output += "(The model ran out of tokens while thinking and didn't write the story yet!)"
        
    return final_output.strip()

if __name__ == "__main__":
    model = "zai-org/GLM-5:novita"
    user_prompt = "Please make me a children's story about the law of gravity."
    
    # Generate the story (using the new higher max_tokens default)
    story = generate_story(user_prompt, model)
    
    print("\n" + story)
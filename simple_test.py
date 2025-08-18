from src.utils.ai import oai, gemini

# Simple test to verify clients work
print("Testing OpenAI client...")
try:
    response = oai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello"}]
    )
    print(f"OpenAI works: {response.choices[0].message.content}")
except Exception as e:
    print(f"OpenAI failed: {e}")

print("\nTesting Gemini client...")
try:
    response = gemini.chat.completions.create(
        model="gemini-2.5-pro",
        messages=[{"role": "user", "content": "Say hello"}]
    )
    print(f"Gemini works: {response.choices[0].message.content}")
except Exception as e:
    print(f"Gemini failed: {e}")

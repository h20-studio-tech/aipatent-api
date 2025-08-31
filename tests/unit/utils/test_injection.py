"""
Test dependency injection without Langfuse dependency.
"""
from src.utils.ai import oai, gemini

# Create a low test function that mimics the pattern
def test_generate_simple(client, model="gpt-4o-mini"):
    """Simple test function with dependency injection."""
    if not client:
        raise ValueError("OpenAI client must be provided")
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Generate a one-sentence field of invention for mRNA vaccines."}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# Test the injection pattern
print("Testing dependency injection pattern...")

print("\n1. Testing with OpenAI client:")
result1 = test_generate_simple(client=oai, model="gpt-4o-mini")
print(f"Result: {result1}")

print("\n2. Testing with Gemini client:")
result2 = test_generate_simple(client=gemini, model="gemini-2.5-pro")
print(f"Result: {result2}")

print("\n3. Testing error handling (no client):")
try:
    result3 = test_generate_simple(client=None)
    print(f"Result: {result3}")
except ValueError as e:
    print(f"Correctly caught error: {e}")

print("\nDependency injection test complete!")

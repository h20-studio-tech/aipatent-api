import os
import sys

def main():
    try:
        # Retrieve the API key from environment
        api_key = os.getenv("OPENAI_API_KEY") or os.environ["OPENAI_API_KEY"]
    except KeyError:
        print("ERROR: OPENAI_API_KEY not set in environment.")
        sys.exit(1)

    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    try:
        # List available models as a connectivity check
        response = client.models.list()
        models = [m.id for m in response.get('data', [])]
        print("✅ OpenAI connectivity OK. First 5 models:", models[:5])
    except Exception as e:
        print("❌ OpenAI connectivity test failed:", str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()

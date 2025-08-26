#!/usr/bin/env python3
"""
Test script to investigate parameter compatibility between OpenAI and Gemini APIs
when using the OpenAI SDK with Gemini's base URL.

This script tests whether Gemini-based OpenAI SDK can gracefully ignore
OpenAI-specific parameters like reasoning_effort.
"""

import os
import asyncio
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# Test configurations
TEST_PROMPT = "What is 2+2? Answer briefly."
TEST_MODEL_OPENAI = "gpt-5"  # Fallback OpenAI model for comparison
TEST_MODEL_GEMINI = "gemini-2.5-flash"  # Gemini model

def test_openai_with_reasoning_effort():
    """Test OpenAI API with reasoning_effort parameter"""
    print("=== Testing OpenAI API with reasoning_effort ===")
    
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": TEST_PROMPT}],
            model=TEST_MODEL_OPENAI,
            reasoning_effort="low"  # OpenAI-specific parameter
        )
        print(f"✅ OpenAI with reasoning_effort: SUCCESS")
        print(f"Response: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"❌ OpenAI with reasoning_effort: FAILED - {e}")
        return False

def test_gemini_without_reasoning_effort():
    """Test Gemini API without reasoning_effort parameter (baseline)"""
    print("\n=== Testing Gemini API without reasoning_effort ===")
    
    try:
        client = OpenAI(
            api_key=os.getenv('GEMINI_API_KEY'),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": TEST_PROMPT}],
            model=TEST_MODEL_GEMINI
            # No reasoning_effort parameter
        )
        print(f"✅ Gemini without reasoning_effort: SUCCESS")
        print(f"Response: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"❌ Gemini without reasoning_effort: FAILED - {e}")
        return False

def test_gemini_with_reasoning_effort():
    """Test Gemini API with reasoning_effort parameter (the key test)"""
    print("\n=== Testing Gemini API with reasoning_effort (KEY TEST) ===")
    
    try:
        client = OpenAI(
            api_key=os.getenv('GEMINI_API_KEY'),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": TEST_PROMPT}],
            model=TEST_MODEL_GEMINI,
            reasoning_effort="low"  # OpenAI-specific parameter - will Gemini ignore it?
        )
        print(f"✅ Gemini with reasoning_effort: SUCCESS - Parameter was ignored!")
        print(f"Response: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"❌ Gemini with reasoning_effort: FAILED - {e}")
        return False

def test_gemini_with_multiple_openai_params():
    """Test Gemini API with multiple OpenAI-specific parameters"""
    print("\n=== Testing Gemini API with multiple OpenAI-specific parameters ===")
    
    try:
        client = OpenAI(
            api_key=os.getenv('GEMINI_API_KEY'),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": TEST_PROMPT}],
            model=TEST_MODEL_GEMINI,
            reasoning_effort="low",  # OpenAI-specific
            # Add other potential OpenAI-specific parameters here if needed
        )
        print(f"✅ Gemini with multiple OpenAI params: SUCCESS - Parameters were ignored!")
        print(f"Response: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"❌ Gemini with multiple OpenAI params: FAILED - {e}")
        return False

async def test_async_gemini_with_reasoning_effort():
    """Test async Gemini API with reasoning_effort parameter"""
    print("\n=== Testing Async Gemini API with reasoning_effort ===")
    
    try:
        client = AsyncOpenAI(
            api_key=os.getenv('GEMINI_API_KEY'),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        response = await client.chat.completions.create(
            messages=[{"role": "user", "content": TEST_PROMPT}],
            model=TEST_MODEL_GEMINI,
            reasoning_effort="low"  # OpenAI-specific parameter
        )
        print(f"✅ Async Gemini with reasoning_effort: SUCCESS - Parameter was ignored!")
        print(f"Response: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"❌ Async Gemini with reasoning_effort: FAILED - {e}")
        return False

def main():
    """Run all compatibility tests"""
    print("🔬 Parameter Compatibility Investigation")
    print("=" * 50)
    
    # Check if required API keys are available
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  Warning: OPENAI_API_KEY not found, skipping OpenAI tests")
    if not os.getenv('GEMINI_API_KEY'):
        print("❌ Error: GEMINI_API_KEY not found, cannot test Gemini")
        return
    
    results = []
    
    # Test OpenAI baseline (if available)
    if os.getenv('OPENAI_API_KEY'):
        results.append(("OpenAI with reasoning_effort", test_openai_with_reasoning_effort()))
    
    # Test Gemini baseline
    results.append(("Gemini without reasoning_effort", test_gemini_without_reasoning_effort()))
    
    # Key test: Gemini with reasoning_effort
    results.append(("Gemini with reasoning_effort", test_gemini_with_reasoning_effort()))
    
    # Test multiple parameters
    results.append(("Gemini with multiple OpenAI params", test_gemini_with_multiple_openai_params()))
    
    # Test async version
    async_result = asyncio.run(test_async_gemini_with_reasoning_effort())
    results.append(("Async Gemini with reasoning_effort", async_result))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SUMMARY OF RESULTS")
    print("=" * 50)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    # Key finding
    gemini_reasoning_success = next((result for name, result in results if "Gemini with reasoning_effort" in name), False)
    
    print("\n🎯 KEY FINDING:")
    if gemini_reasoning_success:
        print("✅ Gemini-based OpenAI SDK CAN gracefully ignore reasoning_effort parameter!")
        print("   This means you can use the same code for both OpenAI and Gemini APIs.")
    else:
        print("❌ Gemini-based OpenAI SDK CANNOT ignore reasoning_effort parameter.")
        print("   You'll need parameter filtering or separate code paths.")

if __name__ == "__main__":
    main()

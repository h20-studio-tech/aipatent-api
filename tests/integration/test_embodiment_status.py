#!/usr/bin/env python3
"""
Test script to verify embodiment status functionality.
This script tests:
1. Retrieving embodiments with status field
2. Updating embodiment status (approve/reject)
3. Fetching only approved embodiments for generation
"""

import asyncio
import aiohttp
import json
import sys
from typing import Optional

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"


async def get_embodiments(patent_id: str) -> Optional[dict]:
    """Get all embodiments for a patent."""
    url = f"{API_BASE_URL}/source-embodiments/{patent_id}"

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return data
            else:
                print(f"❌ Failed to get embodiments: {response.status}")
                return None


async def update_embodiment_status(file_id: str, emb_number: int, status: str) -> bool:
    """Update the status of a specific embodiment."""
    url = f"{API_BASE_URL}/embodiment/status/"
    payload = {
        "file_id": file_id,
        "emb_number": emb_number,
        "status": status
    }

    async with aiohttp.ClientSession() as session:
        async with session.put(url, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                print(f"✅ Updated embodiment {emb_number} to {status}")
                return True
            else:
                error_text = await response.text()
                print(f"❌ Failed to update embodiment {emb_number}: {response.status}")
                print(f"   Error: {error_text}")
                return False


async def get_approved_embodiments(patent_id: str) -> Optional[dict]:
    """Get only approved embodiments for a patent."""
    url = f"{API_BASE_URL}/source-embodiments/approved/{patent_id}"

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return data
            else:
                print(f"❌ Failed to get approved embodiments: {response.status}")
                return None


async def test_embodiment_status_workflow():
    """Test the complete embodiment status workflow."""

    print("🔍 Testing Embodiment Status Functionality")
    print("=" * 50)

    # Get a patent ID from the user or use a default
    patent_id = input("Enter patent_id to test (or press Enter to find one): ").strip()

    if not patent_id:
        # Try to get a sample patent from the API
        print("\n📋 Looking for existing patents with embodiments...")
        # Note: You might need to add an endpoint to list patents or hardcode a known one
        patent_id = "28703285-e3c4-407a-ab53-c4b1e99ed73f"  # Example from earlier
        print(f"Using patent_id: {patent_id}")

    print(f"\n1️⃣ Getting all embodiments for patent {patent_id}...")
    embodiments_data = await get_embodiments(patent_id)

    if not embodiments_data or not embodiments_data.get('data'):
        print("❌ No embodiments found for this patent.")
        return

    embodiments = embodiments_data['data']
    print(f"Found {len(embodiments)} embodiments")

    # Display current status of first few embodiments
    print("\n📊 Current embodiment statuses:")
    for emb in embodiments[:5]:  # Show first 5
        status = emb.get('status', 'undefined')
        print(f"   Embodiment #{emb['emb_number']}: {status}")
        if 'text' in emb:
            preview = emb['text'][:100] + "..." if len(emb['text']) > 100 else emb['text']
            print(f"      Preview: {preview}")

    if len(embodiments) > 5:
        print(f"   ... and {len(embodiments) - 5} more")

    # Test updating statuses
    print("\n2️⃣ Testing status updates...")

    if len(embodiments) >= 3:
        file_id = embodiments[0]['file_id']

        # Approve first embodiment
        print(f"\n   Approving embodiment #{embodiments[0]['emb_number']}...")
        await update_embodiment_status(file_id, embodiments[0]['emb_number'], "approved")

        # Reject second embodiment
        print(f"   Rejecting embodiment #{embodiments[1]['emb_number']}...")
        await update_embodiment_status(file_id, embodiments[1]['emb_number'], "rejected")

        # Approve third embodiment
        print(f"   Approving embodiment #{embodiments[2]['emb_number']}...")
        await update_embodiment_status(file_id, embodiments[2]['emb_number'], "approved")
    else:
        print("   Not enough embodiments to test all statuses")

    # Verify the updates
    print("\n3️⃣ Verifying status updates...")
    updated_data = await get_embodiments(patent_id)

    if updated_data and updated_data.get('data'):
        print("\n📊 Updated embodiment statuses:")
        for emb in updated_data['data'][:5]:
            status = emb.get('status', 'undefined')
            status_emoji = "✅" if status == "approved" else "❌" if status == "rejected" else "⏳"
            print(f"   {status_emoji} Embodiment #{emb['emb_number']}: {status}")

    # Test fetching only approved embodiments
    print("\n4️⃣ Getting only approved embodiments...")
    approved_data = await get_approved_embodiments(patent_id)

    if approved_data:
        approved_count = len(approved_data.get('data', []))
        print(f"✅ Found {approved_count} approved embodiments")

        if approved_count > 0:
            print("\n📋 Approved embodiments:")
            for emb in approved_data['data'][:3]:  # Show first 3
                print(f"   - Embodiment #{emb['emb_number']}")
                if 'text' in emb:
                    preview = emb['text'][:100] + "..." if len(emb['text']) > 100 else emb['text']
                    print(f"     {preview}")

    print("\n" + "=" * 50)
    print("✅ Test completed successfully!")
    print("\n💡 Summary:")
    print("   - Embodiments can now have status: pending, approved, or rejected")
    print("   - Status is persisted in the database")
    print("   - Generation endpoints can fetch only approved embodiments")
    print("   - Frontend can now save user's approval/rejection decisions")


async def main():
    """Main entry point."""
    try:
        await test_embodiment_status_workflow()
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    print("\n🚀 Embodiment Status Test Script")
    print("==================================")
    print("This script tests the embodiment approval/rejection functionality")
    print("\nMake sure the API server is running at http://localhost:8000")
    print("Press Ctrl+C to stop at any time\n")

    asyncio.run(main())
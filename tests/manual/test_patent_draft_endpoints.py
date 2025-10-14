"""
Manual test script for Patent Content Draft endpoints (AIP-1)

This script tests the three new endpoints:
1. PATCH /api/v1/project/patent/{patent_id}/component - Update single component
2. POST /api/v1/project/patent/{patent_id}/save - Save complete draft
3. GET /api/v1/project/patent/{patent_id} - Retrieve draft

Usage:
    uv run python tests/manual/test_patent_draft_endpoints.py
"""

import asyncio
import httpx
import uuid
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8000"
TEST_PATENT_ID = str(uuid.uuid4())

async def test_patch_component():
    """Test updating a single component"""
    print(f"\n{'='*60}")
    print("Test 1: PATCH /api/v1/project/patent/{patent_id}/component")
    print(f"{'='*60}")

    component_data = {
        "component_id": str(uuid.uuid4()),
        "type": "background",
        "title": "Background of the Invention",
        "content": "This invention relates to novel vaccine compositions...",
        "order": 1,
        "trace_id": "test_trace_123",
        "metadata": {
            "model": "gpt-4",
            "prompt_version": "v1"
        }
    }

    async with httpx.AsyncClient() as client:
        response = await client.patch(
            f"{BASE_URL}/api/v1/project/patent/{TEST_PATENT_ID}/component",
            json=component_data,
            timeout=30.0
        )

        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert data["status"] == "success"
        assert data["patent_id"] == TEST_PATENT_ID
        print("✅ Test 1 PASSED")


async def test_patch_multiple_components():
    """Test adding multiple components incrementally"""
    print(f"\n{'='*60}")
    print("Test 2: PATCH multiple components")
    print(f"{'='*60}")

    components = [
        {
            "component_id": str(uuid.uuid4()),
            "type": "field_of_invention",
            "title": "Field of the Invention",
            "content": "This invention is in the field of immunology...",
            "order": 2
        },
        {
            "component_id": str(uuid.uuid4()),
            "type": "summary",
            "title": "Summary of the Invention",
            "content": "The present invention provides compositions...",
            "order": 3
        }
    ]

    async with httpx.AsyncClient() as client:
        for component in components:
            response = await client.patch(
                f"{BASE_URL}/api/v1/project/patent/{TEST_PATENT_ID}/component",
                json=component,
                timeout=30.0
            )
            print(f"Added component '{component['type']}': {response.status_code}")
            assert response.status_code == 200

    print("✅ Test 2 PASSED")


async def test_get_draft():
    """Test retrieving the draft"""
    print(f"\n{'='*60}")
    print("Test 3: GET /api/v1/project/patent/{patent_id}")
    print(f"{'='*60}")

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{BASE_URL}/api/v1/project/patent/{TEST_PATENT_ID}",
            timeout=30.0
        )

        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Draft version: {data.get('version')}")
        print(f"Number of components: {len(data.get('components', []))}")
        print(f"Component types: {[c['type'] for c in data.get('components', [])]}")

        assert response.status_code == 200
        assert data["status"] == "success"
        assert len(data["components"]) == 3  # We added 3 components
        print("✅ Test 3 PASSED")


async def test_save_complete_draft():
    """Test saving complete draft state"""
    print(f"\n{'='*60}")
    print("Test 4: POST /api/v1/project/patent/{patent_id}/save")
    print(f"{'='*60}")

    complete_draft = {
        "components": [
            {
                "id": str(uuid.uuid4()),
                "type": "abstract",
                "title": "Abstract",
                "content": "Novel vaccine compositions are provided...",
                "order": 0,
                "generated_at": datetime.now().isoformat()
            },
            {
                "id": str(uuid.uuid4()),
                "type": "background",
                "title": "Background",
                "content": "Updated background section with more details...",
                "order": 1,
                "generated_at": datetime.now().isoformat()
            },
            {
                "id": str(uuid.uuid4()),
                "type": "claims",
                "title": "Claims",
                "content": "1. A vaccine composition comprising...",
                "order": 4,
                "generated_at": datetime.now().isoformat()
            }
        ]
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/api/v1/project/patent/{TEST_PATENT_ID}/save",
            json=complete_draft,
            timeout=30.0
        )

        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Draft saved - Version: {data.get('version')}")
        print(f"Components count: {data.get('components_count')}")

        assert response.status_code == 200
        assert data["status"] == "success"
        assert data["components_count"] == 3
        assert data["version"] == 2  # Should be version 2 after incremental updates
        print("✅ Test 4 PASSED")


async def test_get_after_save():
    """Verify GET returns the saved draft"""
    print(f"\n{'='*60}")
    print("Test 5: GET after full save (verify version increment)")
    print(f"{'='*60}")

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{BASE_URL}/api/v1/project/patent/{TEST_PATENT_ID}",
            timeout=30.0
        )

        data = response.json()
        print(f"Current version: {data.get('version')}")
        print(f"Number of components: {len(data.get('components', []))}")
        print(f"Component types: {[c['type'] for c in data.get('components', [])]}")

        assert response.status_code == 200
        assert data["version"] == 2
        assert len(data["components"]) == 3
        print("✅ Test 5 PASSED")


async def test_nonexistent_draft():
    """Test GET for non-existent draft"""
    print(f"\n{'='*60}")
    print("Test 6: GET non-existent draft (should return 404)")
    print(f"{'='*60}")

    fake_patent_id = str(uuid.uuid4())

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{BASE_URL}/api/v1/project/patent/{fake_patent_id}",
            timeout=30.0
        )

        print(f"Status Code: {response.status_code}")
        assert response.status_code == 404
        print("✅ Test 6 PASSED")


async def main():
    """Run all tests"""
    print(f"\n{'#'*60}")
    print("Patent Content Draft Endpoints - Manual Test Suite")
    print(f"Testing with patent_id: {TEST_PATENT_ID}")
    print(f"{'#'*60}")

    try:
        # Test incremental component updates
        await test_patch_component()
        await test_patch_multiple_components()
        await test_get_draft()

        # Test full save
        await test_save_complete_draft()
        await test_get_after_save()

        # Test error handling
        await test_nonexistent_draft()

        print(f"\n{'='*60}")
        print("✅ ALL TESTS PASSED!")
        print(f"{'='*60}\n")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""Test script to verify new subsection endpoints are working"""

import requests
import json

BASE_URL = "http://localhost:8000/api/v1/sections/subsections"

# Test data
test_data = {
    "innovation": "Novel IgY antibody therapeutic",
    "technology": "Hyperimmunized egg technology",
    "approach": "Oral passive immunotherapy",
    "antigen": "Enterococcus faecalis",
    "disease": "Alcoholic Liver Disease",
    "additional": "Targeting gut-liver axis"
}

# List of endpoints to test
endpoints = [
    # Summary of Invention subsections
    "target_patient_populations",
    "therapeutic_composition",
    "alternative_embodiments",
    "core_claims",
    # Detailed Description - Disease & Pathology
    "epidemiology_clinical_need",
    # Detailed Description - Therapeutic Formulation
    "hyperimmunized_egg_products",
    "antigenic_targets",
    "production_methods",
    "pharmaceutical_compositions",
    # Detailed Description - Definitions
    "key_terminology",
]

print("Testing new subsection endpoints...")
print("=" * 50)

success_count = 0
fail_count = 0

for endpoint in endpoints:
    url = f"{BASE_URL}/{endpoint}"
    print(f"\nTesting: {endpoint}")

    try:
        response = requests.post(url, json=test_data, timeout=60)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Success - Trace ID: {data.get('trace_id', 'N/A')}")
            print(f"  Preview: {data.get('prediction', '')[:80]}...")
            success_count += 1
        else:
            print(f"✗ Failed - Status: {response.status_code}")
            print(f"  Error: {response.text[:200]}")
            fail_count += 1
    except Exception as e:
        print(f"✗ Exception: {str(e)}")
        fail_count += 1

# Test endpoints with extra parameters
print("\n" + "=" * 50)
print("Testing endpoints with extra parameters...")

# Test disease_specific_overview
disease_specific_data = test_data.copy()
disease_specific_data["disease_name"] = "Alcoholic Liver Disease"
url = f"{BASE_URL}/disease_specific_overview"
print(f"\nTesting: disease_specific_overview")

try:
    response = requests.post(url, json=disease_specific_data, timeout=60)
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Success - Disease: {data.get('disease_name', 'N/A')}")
        print(f"  Preview: {data.get('prediction', '')[:80]}...")
        success_count += 1
    else:
        print(f"✗ Failed - Status: {response.status_code}")
        print(f"  Error: {response.text[:200]}")
        fail_count += 1
except Exception as e:
    print(f"✗ Exception: {str(e)}")
    fail_count += 1

# Test target_in_disease
target_disease_data = test_data.copy()
target_disease_data["target_name"] = "Enterococcus faecalis"
target_disease_data["disease_name"] = "Alcoholic Liver Disease"
url = f"{BASE_URL}/target_in_disease"
print(f"\nTesting: target_in_disease")

try:
    response = requests.post(url, json=target_disease_data, timeout=60)
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Success - Target: {data.get('target_name', 'N/A')} in {data.get('disease_name', 'N/A')}")
        print(f"  Preview: {data.get('prediction', '')[:80]}...")
        success_count += 1
    else:
        print(f"✗ Failed - Status: {response.status_code}")
        print(f"  Error: {response.text[:200]}")
        fail_count += 1
except Exception as e:
    print(f"✗ Exception: {str(e)}")
    fail_count += 1

print("\n" + "=" * 50)
print(f"Testing complete!")
print(f"✓ Success: {success_count}")
print(f"✗ Failed: {fail_count}")
print(f"Total: {success_count + fail_count}")
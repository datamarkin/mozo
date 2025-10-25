"""
Smoke Test for Mozo Model Server

Tests all available models via HTTP to identify which ones work vs which are broken.
This is an integration test - it tests the entire system end-to-end.

Usage:
    1. Start server: mozo start
    2. Run test: python tests/smoke_test.py
    3. Check results: cat test_results.json
"""

import requests
import json
import time
from pathlib import Path

# Configuration
SERVER_URL = "http://localhost:8000"
TEST_IMAGE = "tests/fixtures/test_document.png"  # Test image location
TIMEOUT = 1200  # seconds per model
SKIP_LARGE_MODELS = True  # Skip 7B+ models to save time

# Models that require special parameters
# Can use family name for all variants or specific model_id (family/variant) for variant-specific params
SPECIAL_MODELS = {
    'qwen2.5_vl': {'prompt': 'What is in this image?'},
    'qwen3_vl': {'prompt': 'What is in this image?'},
    'blip_vqa': {'prompt': 'What is in this image?'},
    'florence2': {'prompt': 'Describe this image'},  # For captioning variants
    'florence2/segmentation': {'prompt': 'text'},     # Segmentation needs object-specific prompt
    'stability_inpainting/default': {'prompt': 'a cat'},  # Inpainting prompt
}

# Models that require additional files (e.g., mask for inpainting)
# Format: {model_id: {'file_param_name': 'path/to/file'}}
SPECIAL_FILES = {
    'stability_inpainting/default': {'mask': 'tests/fixtures/test_mask.png'},
}

# Large models to skip (optional)
LARGE_MODELS = ['7b-instruct', '2b-thinking']


def check_server_health():
    """Check if server is running and responsive."""
    try:
        response = requests.get(f"{SERVER_URL}/", timeout=5)
        if response.status_code == 200:
            print(f"✓ Server is healthy: {SERVER_URL}")
            return True
        else:
            print(f"✗ Server returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Cannot connect to server: {e}")
        print(f"  Make sure server is running: mozo start")
        return False


def get_all_models():
    """Fetch all available models from /models endpoint."""
    try:
        response = requests.get(f"{SERVER_URL}/models", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Found {len(data)} model families")

            # Count total variants
            total_variants = sum(len(info['variants']) for info in data.values())
            print(f"✓ Found {total_variants} total variants")
            return data
        else:
            print(f"✗ Failed to fetch models: HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"✗ Error fetching models: {e}")
        return None


def test_model(family, variant, test_image_path):
    """
    Test a single model variant via HTTP prediction.

    Returns:
        tuple: (success: bool, message: str, response_data: dict or None)
    """
    model_id = f"{family}/{variant}"

    # Skip large models if configured
    if SKIP_LARGE_MODELS and variant in LARGE_MODELS:
        return False, f"Skipped (large model)", None

    # Check if test image exists
    if not test_image_path.exists():
        return False, f"Test image not found: {test_image_path}", None

    try:
        # Prepare request parameters
        params = {}
        # Check for variant-specific params first, then family-level params
        if model_id in SPECIAL_MODELS:
            params.update(SPECIAL_MODELS[model_id])
        elif family in SPECIAL_MODELS:
            params.update(SPECIAL_MODELS[family])

        # Prepare files dict
        files_dict = {}

        # Open main image file
        with open(test_image_path, "rb") as f:
            files_dict["file"] = f

            # Check if model needs additional files
            if model_id in SPECIAL_FILES:
                special_file_handles = []
                try:
                    for file_param, file_path in SPECIAL_FILES[model_id].items():
                        file_handle = open(file_path, "rb")
                        special_file_handles.append(file_handle)
                        files_dict[file_param] = file_handle

                    # Make prediction request with all files
                    response = requests.post(
                        f"{SERVER_URL}/predict/{family}/{variant}",
                        files=files_dict,
                        params=params,
                        timeout=TIMEOUT
                    )
                finally:
                    # Close additional file handles
                    for handle in special_file_handles:
                        handle.close()
            else:
                # Make prediction request with just main file
                response = requests.post(
                    f"{SERVER_URL}/predict/{family}/{variant}",
                    files=files_dict,
                    params=params,
                    timeout=TIMEOUT
                )

        # Check response
        if response.status_code == 200:
            # Check content type to determine if it's an image
            content_type = response.headers.get('content-type', '')

            if 'image' in content_type:
                # Save image responses for visual verification
                if model_id == 'stability_inpainting/default':
                    output_path = Path(__file__).parent / 'fixtures' / 'inpaint_result.png'
                    try:
                        with open(output_path, 'wb') as f:
                            f.write(response.content)
                        return True, "Success (image saved)", None
                    except Exception as e:
                        return True, f"Success (save failed: {e})", None
                else:
                    return True, "Success (image response)", None
            else:
                # Try to parse as JSON
                try:
                    data = response.json()
                    return True, "Success", data
                except:
                    # Non-JSON, non-image response
                    return True, "Success (non-JSON response)", None
        else:
            try:
                error_detail = response.json().get('detail', 'Unknown error')
            except:
                error_detail = response.text[:100]
            return False, f"HTTP {response.status_code}: {error_detail}", None

    except requests.exceptions.Timeout:
        return False, f"Timeout (>{TIMEOUT}s)", None
    except Exception as e:
        error_msg = str(e)[:100]  # Truncate long errors
        return False, f"Exception: {error_msg}", None


def run_smoke_test():
    """Run smoke test on all available models."""
    print("="*70)
    print("MOZO SMOKE TEST")
    print("="*70)
    print()

    # Check server health
    if not check_server_health():
        return None
    print()

    # Get all models
    all_models = get_all_models()
    if not all_models:
        return None
    print()

    # Find test image
    test_image_path = Path(TEST_IMAGE)
    if not test_image_path.exists():
        print(f"✗ Test image not found: {TEST_IMAGE}")
        print(f"  Current directory: {Path.cwd()}")
        return None
    print(f"✓ Using test image: {test_image_path}")
    print()

    # Test all models
    results = {
        "working": [],
        "broken": [],
        "skipped": [],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_families": len(all_models),
        "total_variants": 0
    }

    print("="*70)
    print("TESTING MODELS")
    print("="*70)

    for family, info in sorted(all_models.items()):
        variants = info.get('variants', [])
        results["total_variants"] += len(variants)

        print(f"\n{family} ({len(variants)} variants):")
        print("-" * 70)

        for variant in variants:
            model_id = f"{family}/{variant}"

            # Test the model
            success, message, response_data = test_model(family, variant, test_image_path)

            # Record result
            result_entry = {
                "model_id": model_id,
                "family": family,
                "variant": variant,
                "message": message
            }

            if "Skipped" in message:
                results["skipped"].append(result_entry)
                print(f"  ⊘ {variant:30s} - {message}")
            elif success:
                results["working"].append(result_entry)
                print(f"  ✓ {variant:30s} - {message}")
            else:
                results["broken"].append(result_entry)
                print(f"  ✗ {variant:30s} - {message}")

    return results


def print_summary(results):
    """Print test summary."""
    if not results:
        return

    total = results["total_variants"]
    working = len(results["working"])
    broken = len(results["broken"])
    skipped = len(results["skipped"])

    print()
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Total Variants:  {total}")
    print(f"✓ Working:       {working:3d} ({working*100//total:2d}%)")
    print(f"✗ Broken:        {broken:3d} ({broken*100//total:2d}%)")
    print(f"⊘ Skipped:       {skipped:3d} ({skipped*100//total:2d}%)")
    print("="*70)

    # Show broken models
    if results["broken"]:
        print()
        print("BROKEN MODELS:")
        print("-" * 70)
        for entry in results["broken"]:
            print(f"  {entry['model_id']:40s} - {entry['message']}")

    # Show skipped models
    if results["skipped"]:
        print()
        print("SKIPPED MODELS:")
        print("-" * 70)
        for entry in results["skipped"]:
            print(f"  {entry['model_id']:40s} - {entry['message']}")


def save_results(results):
    """Save results to JSON file."""
    if not results:
        return

    output_file = "test_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print()
    print(f"✓ Results saved to: {output_file}")


def main():
    """Main entry point."""
    results = run_smoke_test()

    if results:
        print_summary(results)
        save_results(results)

        # Exit with error code if any models are broken
        if results["broken"]:
            print()
            print(f"⚠️  WARNING: {len(results['broken'])} model(s) are broken!")
            return 1
        else:
            print()
            print("✓ All tested models are working!")
            return 0
    else:
        print()
        print("✗ Smoke test failed to complete")
        return 1


if __name__ == "__main__":
    exit(main())

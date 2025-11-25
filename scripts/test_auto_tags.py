"""Test script for auto-tag extraction functionality."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import httpx
from services.preprocessor import preprocess_text


async def main():
    """Test auto-tag extraction."""
    print("=" * 70)
    print("Testing Auto-Tag Extraction")
    print("=" * 70)
    
    # Test cases
    test_cases = [
        {
            "content": "My flight to London is at 3 PM tomorrow. I love traveling.",
            "user_tags": ["travel", "london"],
            "expected_auto_tags": ["flight", "london", "pm", "tomorrow", "love", "traveling"]
        },
        {
            "content": "I learned Python programming in 2023. It's a versatile language for data science.",
            "user_tags": [],
            "expected_auto_tags": ["learned", "python", "programming", "versatile", "language", "data", "science"]
        },
        {
            "content": "Meeting with John at 2pm to discuss the project deadline.",
            "user_tags": ["work"],
            "expected_auto_tags": ["meeting", "john", "pm", "discuss", "project", "deadline"]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"  Content: {test_case['content'][:60]}...")
        print(f"  User Tags: {test_case['user_tags']}")
        
        # Preprocess
        preprocessed = await preprocess_text(test_case['content'])
        extracted_tags = preprocessed.get('extracted_tags', [])
        
        print(f"  Extracted Tags: {extracted_tags}")
        
        # Simulate tag merging (as done in memory creation)
        user_tags = test_case['user_tags']
        all_tags = list(user_tags)
        for tag in extracted_tags:
            tag_lower = tag.lower().strip()
            if tag_lower not in [t.lower().strip() for t in all_tags]:
                all_tags.append(tag)
        
        final_tags = [tag.strip() for tag in all_tags if tag.strip()]
        print(f"  Final Tags (merged): {final_tags}")
        
        # Verify
        if extracted_tags:
            print(f"  [OK] Tags extracted successfully")
        else:
            print(f"  [WARNING] No tags extracted")
    
    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())


#!/usr/bin/env python3
"""Test script to explore the Google GenerativeAI client API structure."""

import os
import sys
import google.generativeai as genai
from google.generativeai import client as genai_client

# Try to get API key from config or environment
try:
    from src.config import config
    api_key = config.google_api_key
    print("Using API key from config module")
except Exception as e:
    print(f"Could not load config: {e}")
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        print("ERROR: GOOGLE_API_KEY environment variable not set")
        sys.exit(1)

# Configure the API
genai.configure(api_key=api_key)

print("=== Exploring Google GenerativeAI Client API ===\n")

# 1. Get the default generative client
print("1. Getting default generative client:")
client = genai_client.get_default_generative_client()
print(f"   Type: {type(client)}")
print(f"   Class: {client.__class__.__name__}")

# 2. Explore client attributes
print("\n2. Client attributes:")
attrs = [attr for attr in dir(client) if not attr.startswith('_')]
for attr in attrs[:10]:  # Show first 10 attributes
    print(f"   - {attr}")
print(f"   ... and {len(attrs) - 10} more attributes")

# 3. Check for specific attributes we're interested in
print("\n3. Checking for specific attributes:")
important_attrs = ['models', 'files', 'generate_content']
for attr in important_attrs:
    has_attr = hasattr(client, attr)
    print(f"   - Has '{attr}': {has_attr}")
    if has_attr:
        attr_obj = getattr(client, attr)
        print(f"     Type: {type(attr_obj)}")

# 4. Explore the models attribute
print("\n4. Exploring client.models:")
if hasattr(client, 'models'):
    models = client.models
    print(f"   Type: {type(models)}")
    print(f"   Class: {models.__class__.__name__}")
    
    # Check what methods are available on models
    model_methods = [m for m in dir(models) if not m.startswith('_') and callable(getattr(models, m))]
    print(f"   Methods available:")
    for method in model_methods[:5]:
        print(f"     - {method}")
    if len(model_methods) > 5:
        print(f"     ... and {len(model_methods) - 5} more methods")

# 5. Test a simple generate_content call
print("\n5. Testing generate_content call:")
try:
    # First, let's check if we can call it directly
    print("   Attempting to call client.models.generate_content...")
    
    # Simple test prompt
    response = client.models.generate_content(
        model="models/gemini-2.0-flash-exp",
        contents=[{
            "role": "user",
            "parts": [{"text": "Say 'Hello, API test!' and nothing else."}]
        }]
    )
    
    print("   SUCCESS! Response received.")
    print(f"   Response type: {type(response)}")
    
    # Try to access response content
    if hasattr(response, 'text'):
        print(f"   Response text: {response.text}")
    elif hasattr(response, 'candidates'):
        print(f"   Has candidates: {len(response.candidates) if response.candidates else 0}")
        if response.candidates and response.candidates[0].content.parts:
            print(f"   First part text: {response.candidates[0].content.parts[0].text}")
    
except Exception as e:
    print(f"   ERROR: {type(e).__name__}: {str(e)}")

# 6. Test with config parameter
print("\n6. Testing generate_content with config parameter:")
try:
    import google.generativeai.types as genai_types
    
    response = client.models.generate_content(
        model="models/gemini-2.0-flash-exp",
        config=genai_types.GenerateContentConfig(
            temperature=0.5,
            system_instruction="You are a helpful assistant."
        ),
        contents=[{
            "role": "user",
            "parts": [{"text": "Say 'Config test passed!' and nothing else."}]
        }]
    )
    
    print("   SUCCESS! Response with config received.")
    if response.candidates and response.candidates[0].content.parts:
        print(f"   Response: {response.candidates[0].content.parts[0].text}")
    
except Exception as e:
    print(f"   ERROR: {type(e).__name__}: {str(e)}")

# 7. Check for alternative methods
print("\n7. Checking for alternative content generation methods:")
alternative_methods = ['generate_content_stream', 'count_tokens', 'embed_content']
for method in alternative_methods:
    if hasattr(client.models, method):
        print(f"   - Has '{method}': True")
    else:
        print(f"   - Has '{method}': False")

print("\n=== Test Complete ===")
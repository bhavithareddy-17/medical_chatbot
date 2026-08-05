#!/usr/bin/env python3
# check_models.py — Run this to see which Gemini models your API key supports
import os
from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("❌ No API key found in .env")
    exit(1)

genai.configure(api_key=api_key)

print("\n✅ API Key loaded. Fetching available models...\n")
print("=" * 60)

flash_models = []
for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(f"  ✓  {m.name}")
        if "flash" in m.name.lower():
            flash_models.append(m.name)

print("=" * 60)
print(f"\n💡 Recommended model to use in config.py:")
if flash_models:
    # Strip "models/" prefix for LangChain
    best = flash_models[0].replace("models/", "")
    print(f"   LLM_MODEL = \"{best}\"")
else:
    print("   No flash models found — check your API key")
print()
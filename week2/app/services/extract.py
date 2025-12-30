from __future__ import annotations

import os
import re
from typing import List
import json
import logging
from typing import Any
from ollama import chat
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

BULLET_PREFIX_PATTERN = re.compile(r"^\s*([-*•]|\d+\.)\s+")
KEYWORD_PREFIXES = (
    "todo:",
    "action:",
    "next:",
)


def _is_action_line(line: str) -> bool:
    stripped = line.strip().lower()
    if not stripped:
        return False
    if BULLET_PREFIX_PATTERN.match(stripped):
        return True
    if any(stripped.startswith(prefix) for prefix in KEYWORD_PREFIXES):
        return True
    if "[ ]" in stripped or "[todo]" in stripped:
        return True
    return False


def extract_action_items(text: str) -> List[str]:
    lines = text.splitlines()
    extracted: List[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if _is_action_line(line):
            cleaned = BULLET_PREFIX_PATTERN.sub("", line)
            cleaned = cleaned.strip()
            # Trim common checkbox markers
            cleaned = cleaned.removeprefix("[ ]").strip()
            cleaned = cleaned.removeprefix("[todo]").strip()
            extracted.append(cleaned)
    # Fallback: if nothing matched, heuristically split into sentences and pick imperative-like ones
    if not extracted:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        for sentence in sentences:
            s = sentence.strip()
            if not s:
                continue
            if _looks_imperative(s):
                extracted.append(s)
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: List[str] = []
    for item in extracted:
        lowered = item.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        unique.append(item)
    return unique

def _looks_imperative(sentence: str) -> bool:
    words = re.findall(r"[A-Za-z']+", sentence)
    if not words:
        return False
    first = words[0]
    # Crude heuristic: treat these as imperative starters
    imperative_starters = {
        "add",
        "create",
        "implement",
        "fix",
        "update",
        "write",
        "check",
        "verify",
        "refactor",
        "document",
        "design",
        "investigate",
    }
    return first.lower() in imperative_starters


class ActionItemsResponse(BaseModel):
    """Pydantic model for LLM response structure"""
    action_items: List[str]

def extract_action_items_llm(text: str) -> List[str]:
    """
    Extract action items from text using LLM, with identical signature and behavior
    as the original extract_action_items function.
    
    Args:
        text: Input text containing potential action items
        
    Returns:
        List of cleaned action items, deduplicated while preserving order
    """
    try:
        # Create a precise prompt reflecting the original function's exact behavior
        prompt = f"""
You are an expert in extracting action items from text. Extract all action items following these EXACT rules:

1. Identify action items by:
   - Bullet point prefixes (-, *, •, or numbered lists like "1.")
   - Keyword prefixes: "todo:", "action:", "next:"
   - Checkbox markers: "[ ]" or "[todo]"
   - Imperative sentences starting with: add, create, implement, fix, update, write, 
     check, verify, refactor, document, design, investigate

2. For each action item:
   - Remove bullet point prefixes
   - Remove checkbox markers ([ ] or [todo])
   - Do NOT remove keyword prefixes (todo:, action:, next:)
   - Trim extra whitespace

3. Final processing:
   - Deduplicate items (keep only first occurrence of each unique item)
   - Preserve the original order of first occurrences
   - Return ONLY a JSON object with an "action_items" field containing the cleaned list

Text to analyze:
{text}

IMPORTANT: Return ONLY the JSON object with the "action_items" field, no explanations or additional text.
        """
        
        # Call Ollama with the structured prompt and JSON format
        response = chat(
            model='qwen3:4b',
            messages=[{'role': 'user', 'content': prompt}],
            format=ActionItemsResponse.model_json_schema(),
            options={'temperature': 0.}
        )
        
        # Parse the response content into our model
        result = ActionItemsResponse.model_validate_json(response['message']['content'])
        
        # Deduplicate while preserving order (exactly matching original function behavior)
        seen = set()
        unique_items = []
        for item in result.action_items:
            lowered = item.lower()
            if lowered not in seen:
                seen.add(lowered)
                unique_items.append(item)
                
        return unique_items
        
    except Exception as e:
        # Log the error but return empty list to match original function behavior
        logging.error(f"LLM extraction failed: {str(e)}")
        return []
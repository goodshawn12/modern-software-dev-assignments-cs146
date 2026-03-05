from __future__ import annotations

import os
import re
import json
from typing import Any, List
from ollama import chat
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class ActionItemsResponse(BaseModel):
    action_items: List[str]

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


def extract_action_items_llm(text: str) -> List[str]:
    """
    Extracts action items from the given text using an LLM (Ollama).
    Uses structured output to ensure a clean list of strings is returned.
    """
    
    prompt = f"""
    Extract all actionable tasks and to-do items from the following text. 
    Focus on specific, clear instructions or commitments.
    Return only the action items.

    Text:
    {text}
    """
    
    try:
        response = chat(
            model="llama3.1:8b",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts action items from notes. You must return a JSON object with a key 'action_items' containing a list of strings."},
                {"role": "user", "content": prompt},
            ],
            format=ActionItemsResponse.model_json_schema(),
        )
        print(ActionItemsResponse.model_json_schema())
        # Parse the structured response
        data = ActionItemsResponse.model_validate_json(response.message.content)
        return data.action_items
    except Exception as e:
        # Fallback or log error (for now just returning empty or print for visibility in development)
        print(f"Error calling Ollama: {e}")
        return []

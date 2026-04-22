"""
Robust JSON extraction from VLM outputs.

VLMs frequently wrap JSON in markdown blocks, add preamble text,
or produce malformed JSON. This parser handles all common cases.
"""

import json
import re


def parse_response(raw_text: str) -> dict:
    """Parse VLM response to extract decision and explanation.

    Returns:
        {"decision": bool|None, "explanation": str, "parse_error": bool}
    """
    if not raw_text or not raw_text.strip():
        return {"decision": None, "explanation": "", "parse_error": True}

    text = raw_text.strip()

    # Try 1: Direct JSON parse
    try:
        obj = json.loads(text)
        return _validate(obj)
    except json.JSONDecodeError:
        pass

    # Try 2: Extract from markdown code block
    md_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if md_match:
        try:
            obj = json.loads(md_match.group(1).strip())
            return _validate(obj)
        except json.JSONDecodeError:
            pass

    # Try 3: Find JSON object in text
    json_match = re.search(r'\{[^{}]*"decision"\s*:\s*(true|false)[^{}]*\}',
                           text, re.IGNORECASE | re.DOTALL)
    if json_match:
        try:
            obj = json.loads(json_match.group(0))
            return _validate(obj)
        except json.JSONDecodeError:
            pass

    # Try 4: Find JSON-like structure with explanation
    json_match2 = re.search(
        r'\{\s*"explanation"\s*:\s*"(.*?)"\s*,\s*"decision"\s*:\s*(true|false)\s*\}',
        text, re.IGNORECASE | re.DOTALL)
    if json_match2:
        explanation = json_match2.group(1)
        decision = json_match2.group(2).lower() == "true"
        return {"decision": decision, "explanation": explanation, "parse_error": False}

    # Try 5: Fallback — scan for true/false keywords
    text_lower = text.lower()
    if '"decision": true' in text_lower or '"decision":true' in text_lower:
        decision = True
    elif '"decision": false' in text_lower or '"decision":false' in text_lower:
        decision = False
    elif text_lower.rstrip().endswith("true"):
        decision = True
    elif text_lower.rstrip().endswith("false"):
        decision = False
    else:
        return {"decision": None, "explanation": text[:200], "parse_error": True}

    return {"decision": decision, "explanation": text[:200], "parse_error": True}


def _validate(obj: dict) -> dict:
    """Validate parsed JSON has required fields."""
    decision = obj.get("decision")
    explanation = obj.get("explanation", "")

    if isinstance(decision, bool):
        return {"decision": decision, "explanation": explanation, "parse_error": False}
    if isinstance(decision, str):
        decision = decision.lower() in ("true", "yes", "1")
        return {"decision": decision, "explanation": explanation, "parse_error": False}

    return {"decision": None, "explanation": explanation, "parse_error": True}

"""
VLM Jury prompt templates following Ruck et al. (2026).

Two evaluation modes:
  1. Paired: original clear + augmented side-by-side → binary decision
  2. Baseline: single ACDC real weather image → binary decision (ceiling)
"""

# Condition-specific visual cues for judges
CONDITION_GUIDANCE = {
    "rain": (
        "Rain augmentation should show: visible rain streaks or precipitation, "
        "wet or reflective surfaces, overcast or dark sky, puddles or water accumulation, "
        "overall grey/blue color tone shift."
    ),
    "snow": (
        "Snow augmentation should show: white snow coverage on surfaces, "
        "falling snow particles, overcast grey sky, reduced color saturation, "
        "frost or ice effects on surfaces."
    ),
    "fog": (
        "Fog augmentation should show: reduced visibility especially at distance, "
        "atmospheric haze or white-out effect, muted/washed-out colors, "
        "obscured far objects, depth-dependent visibility loss."
    ),
    "night": (
        "Night augmentation should show: dark ambient lighting, "
        "artificial light sources (floodlights, lamps), visible shadows from lights, "
        "reduced color saturation, overall dark blue/black tone."
    ),
}

SYSTEM_PROMPT = (
    "You are an expert image quality assessor evaluating synthetic weather "
    "augmentation for construction site images. You make binary decisions "
    "based on visual evidence. Be objective and concise."
)

PAIRED_PROMPT_TEMPLATE = """Evaluate this side-by-side image pair of a construction site.

The LEFT half shows the ORIGINAL clear-weather image.
The RIGHT half shows the AUGMENTED image with synthetic {condition} effects applied.

Assess TWO criteria:

1. **Condition Realism**: Does the {condition} effect in the right image look realistic?
   {guidance}

2. **Semantic Preservation**: Apart from the weather change, is the scene content preserved?
   Objects, structures, spatial layout should remain the same. Only weather-related changes
   (e.g., reduced visibility in fog, wet surfaces in rain) are acceptable.

Both criteria must be satisfied for a positive decision.

Respond with ONLY a JSON object (no other text):
{{"explanation": "<brief 1-2 sentence reasoning>", "decision": true/false}}"""

BASELINE_PROMPT_TEMPLATE = """Evaluate this single image.

Does this image convincingly depict realistic {condition} conditions?
{guidance}

Consider: does the {condition} effect look natural and believable?

Respond with ONLY a JSON object (no other text):
{{"explanation": "<brief 1-2 sentence reasoning>", "decision": true/false}}"""


def get_paired_prompt(condition: str) -> str:
    """Get prompt for paired (original + augmented) evaluation."""
    cond_key = _normalize_condition(condition)
    guidance = CONDITION_GUIDANCE[cond_key]
    return PAIRED_PROMPT_TEMPLATE.format(condition=cond_key, guidance=guidance)


def get_baseline_prompt(condition: str) -> str:
    """Get prompt for baseline (single ACDC image) evaluation."""
    cond_key = _normalize_condition(condition)
    guidance = CONDITION_GUIDANCE[cond_key]
    return BASELINE_PROMPT_TEMPLATE.format(condition=cond_key, guidance=guidance)


def _normalize_condition(condition: str) -> str:
    """Map condition keys to standard names."""
    cond = condition.lower()
    if "rain" in cond:
        return "rain"
    if "snow" in cond:
        return "snow"
    if "fog" in cond:
        return "fog"
    if "night" in cond:
        return "night"
    return cond

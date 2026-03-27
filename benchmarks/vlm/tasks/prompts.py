"""
Centralized prompts — exact copies from the author's implementation.
"""

SYSTEM_PROMPT_INSPECTOR = (
    "You are a construction site safety inspector. You are responsible for "
    "viewing the given image and give helpful, detailed, and polite answers "
    "to your supervisor. You only answer questions that are asked by the "
    "supervisor and in the exact way as requested."
)

# ── Description task ─────────────────────────────────────────────────────

DESCRIPTION_PROMPT = "Describe this construction site image in detail."

DESCRIPTION_PROMPT_DETAILED = (
    "Describe this construction site image in detail. Include the number, "
    "location, activities of people, equipment, stockpiles. Do not make "
    "assumptions, be concise, one paragraph, no greetings."
)

# ── VQA Safety (4 rules, 5-shot) ────────────────────────────────────────

FEW_SHOT_INTRO = (
    "You will be asked to read the image and identify violations of safety "
    "rules that appears in the image. You also need to provide a short "
    "reasoning and bounding boxs showing the location of the violation.\n\n"
    "I will give you five examples to show you how your answer should be "
    "formatted and what your reasoning should include."
)

VQA_SAFETY_PROMPT = (
    'Please read the image and identify if there are violations of the '
    'following four safety rules in the image, do not include violations '
    'that do not exist in your answer, assume no violation if the visual '
    'information is not enough to make a judgement:\n\n'
    '1. Use of basic PPE when on foot at construction sites. Machine '
    'operators do not need PPE. (hard hats, properly worn clothes covering '
    'shoulders and legs, shoes that can cover toes, high-visibility '
    'retroreflective vests at night, face shield or safety glasses when '
    'cutting, welding, grinding, or drilling).\n\n'
    '2. Use of safety harness when working from a height of three meters '
    'and the edges are without any edge protection.\n\n'
    '3. Adoption of edge protection or edge warning including guardrails, '
    'fences, for underground projects three meters in depth with steep '
    'retaining wall and for human to stand.\n\n'
    '4. Appearance of worker in the blind spots of the operator and within '
    'the operation radius of excavators in operation, or excavators with '
    'operators inside.\n\n'
    'Your answer should be in the format of {"id of the safety rule": '
    '{"reason": one or two sentences explaining who violate the rule in '
    'the image and the specific reason, "bounding_box": [the location of '
    'violation in the image x_min, y_min, x_max, y_max in 0-1 normalized '
    'space]}}.\n\n'
    'Return {"0": "No violations"} if you find no violation in the image.'
)

# Few-shot expected outputs (from author)
FEWSHOT_ANSWERS = {
    '0000001': '{"0": "No violations"}',
    '0000007': (
        '{"1": {"reason": "Multiple workers not wearing hard hats nor '
        'high-visibility vests working at night.", "bounding_box": '
        '[0.14, 0.09, 1.0, 0.66]}, "3": {"reason": "Opening not protected '
        'on both the left and the right of the images.", "bounding_box": '
        '[0.0, 0.53, 0.46, 0.99]}}'
    ),
    '0000019': (
        '{"1": {"reason": "Worker with a black cap and white shirt on the '
        'left is not wearing a hard hat.", "bounding_box": '
        '[0.27, 0.47, 0.42, 0.68]}}'
    ),
    '0000327': (
        '{"4": {"reason": "The worker holding an umbrella is too close to '
        'the excavator is operation.", "bounding_box": '
        '[0.25, 0.28, 0.95, 0.76]}}'
    ),
    '0004235': (
        '{"1": {"reason": "None of the workers wear a hard hat. The worker '
        'on the ground level in the middle is not wearing his shirt properly '
        'while the worker on top of the scaffold wears a sleeveless shirt.", '
        '"bounding_box": [0.02, 0.01, 0.44, 0.47]}, "2": {"reason": "The '
        'worker standing on the scaffold does not have a safety harness.", '
        '"bounding_box": [0.01, 0.04, 0.25, 0.48]}}'
    ),
}

FEWSHOT_IDS = ['0000001', '0000007', '0000019', '0000327', '0004235']

# ── VQA Rule 1 (binary Yes/No) ──────────────────────────────────────────

VQA_RULE1_PROMPT = (
    'Rule 1: Use of basic PPE when on foot at construction sites (hard hats, '
    'properly worn clothes covering shoulders and legs, shoes that can cover '
    'toes, high-visibility retroreflective vests at night, face shield or '
    'safety glasses when cutting, welding, grinding, or drilling).\n\n'
    'Based on the image, is there a violation of Rule 1? Answer only Yes or No.'
)

# ── Object Detection ────────────────────────────────────────────────────

OBJECT_DETECTION_PROMPTS = {
    'excavator': (
        'Detect all instances of excavator in the image. '
        'Return location as [x_min, y_min, x_max, y_max] in 0-1 normalized space. '
        'Return ["None"] if not found.'
    ),
    'workers': (
        'Detect all instances of worker with white hard hat in the image. '
        'Return location as [x_min, y_min, x_max, y_max] in 0-1 normalized space. '
        'Return ["None"] if not found.'
    ),
    'rebars': (
        'Detect all instances of rebar in the image. '
        'Return location as [x_min, y_min, x_max, y_max] in 0-1 normalized space. '
        'Return ["None"] if not found.'
    ),
}

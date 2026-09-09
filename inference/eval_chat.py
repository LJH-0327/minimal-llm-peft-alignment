import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

sys.path.append(str(ROOT_DIR))

import json

from loader import load_model
from generate import generate_response

from config import (
    MODEL_PATH,
    OUTPUT_ADAPTER_PATH,
    RESULT_DIR,
    PROMPT_PATH
) 


RESULT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = RESULT_DIR / "eval_result_lora.json"

model, tokenizer = load_model(
    str(MODEL_PATH),
    str(OUTPUT_ADAPTER_PATH)
)


with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    prompts = json.load(f)


results = []

for item in prompts:

    messages = [
        {
            "role": "user",
            "content": item["instruction"]
        }
    ]

    response = generate_response(
        model,
        tokenizer,
        messages,
        do_sample=False
    )

    results.append(
        {
            "instruction": item["instruction"],
            "response": response
        }
    )

    print("=" * 60)
    print(f"User: {item['instruction']}")
    print()
    print(f"Assistant: {response}")


with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        ensure_ascii=False,
        indent=2
    )


print(f"\nResults saved to {OUTPUT_PATH}")

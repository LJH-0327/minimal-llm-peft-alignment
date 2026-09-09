import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

sys.path.append(str(ROOT_DIR))

from loader import load_model
from generate import generate_response

from config import (
    MODEL_NAME,
    DATASET_NAME,
    MODEL_PATH,
    OUTPUT_ADAPTER_PATH,
)

print(f"Model: {MODEL_NAME}")
print(f"Adapter: {DATASET_NAME}")

model, tokenizer = load_model(
    str(MODEL_PATH),
    str(OUTPUT_ADAPTER_PATH)
)

messages = []


while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Chat ended.")
        break

    messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    response = generate_response(
        model,
        tokenizer,
        messages,
        do_sample=False
    )

    messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )

    print(f"Assistant: {response}")


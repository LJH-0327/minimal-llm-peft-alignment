MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct"


LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


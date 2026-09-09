import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

sys.path.append(str(ROOT_DIR))

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

from peft import PeftModel


def load_model(model_path, adapter_path=None):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map="auto"
    )

    if adapter_path is not None:
        model = PeftModel.from_pretrained(
            model, 
            adapter_path
        )

    model.eval()

    return model, tokenizer
    
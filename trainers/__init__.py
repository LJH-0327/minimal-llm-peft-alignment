from config import MODE

from .sft import build_trainer as build_sft_trainer
from .dpo import build_trainer as build_dpo_trainer

TRAINER_REGISTRY = {
    "sft": build_sft_trainer,
    "dpo": build_dpo_trainer,
}

def build_trainer(*args, **kwargs):
    try:
        builder = TRAINER_REGISTRY[MODE]
    except KeyError:
        raise ValueError(f"Unsupported mode: {MODE}")

    return builder(*args, **kwargs)


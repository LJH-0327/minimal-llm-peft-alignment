from pathlib import Path

from .runtime import MODE
from .model import MODEL_NAME
from .dataset import OUTPUT_DATASET_NAME, SFT_DATASET_NAME

ROOT_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = ROOT_DIR / "models"

DATASETS_DIR = ROOT_DIR / "datasets"
RAW_DATASETS_DIR = DATASETS_DIR / "raw"
 
ADAPTERS_DIR = ROOT_DIR / "adapters"
RESULTS_DIR = ROOT_DIR / "results"

MODEL_PATH = MODELS_DIR / MODEL_NAME
PROMPT_PATH = DATASETS_DIR / "eval_prompts" / "dpo.json"
RAW_DATASET_PATH = RAW_DATASETS_DIR / OUTPUT_DATASET_NAME


SFT_ADAPTER_PATH = (
    ADAPTERS_DIR
    / "sft"
    / MODEL_NAME
    / SFT_DATASET_NAME
)

if MODE == "sft":
    OUTPUT_ADAPTER_PATH = SFT_ADAPTER_PATH
    RESULT_DIR = (
        RESULTS_DIR
        / "sft"
        / MODEL_NAME
        / OUTPUT_DATASET_NAME
    )

elif MODE in {"dpo", "ppo", "simpo", "kto"}:
    OUTPUT_ADAPTER_PATH = (
        ADAPTERS_DIR
        / MODE
        / MODEL_NAME
        / OUTPUT_DATASET_NAME
        / SFT_DATASET_NAME
    )
    RESULT_DIR = (
        RESULTS_DIR
        / MODE
        / MODEL_NAME
        / OUTPUT_DATASET_NAME
        / SFT_DATASET_NAME
    )

else:
    raise ValueError(f"Unsupported MODE: {MODE}")


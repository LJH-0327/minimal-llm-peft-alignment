# trainers/base.py

from config import *


def build_common_training_args() -> dict:
    """
    Build common training arguments shared by all trainers.
    """

    return {
        "output_dir": str(OUTPUT_ADAPTER_PATH),

        # Training
        "learning_rate": LEARNING_RATE,
        "num_train_epochs": NUM_TRAIN_EPOCHS,
        "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
        "per_device_eval_batch_size": PER_DEVICE_EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "warmup_steps": WARMUP_STEPS,
        "lr_scheduler_type": LR_SCHEDULE_TYPE,
        "weight_decay": WEIGHT_DECAY,
        "optim": OPTIM,


        "max_length" : MAX_LENGTH,

        "gradient_checkpointing" : GRADIENT_CHECKPOINTING,

        # Logging
        "logging_strategy": LOGGING_STRATEGY,
        "logging_steps": LOGGING_STEPS,

        # Evaluation
        "eval_strategy": EVAL_STRATEGY,
        "eval_steps": EVAL_STEPS,

        # Saving
        "save_strategy": SAVE_STRATEGY,
        "save_steps": SAVE_STEPS,
        "save_total_limit": SAVE_TOTAL_LIMIT,

        # Best checkpoint
        "load_best_model_at_end": LOAD_BEST_MODEL_AT_END,
        "metric_for_best_model": METRIC_FOR_BEST_MODEL,
        "greater_is_better": GREATER_IS_BETTER,

        # Runtime
        "seed": SEED,
        "fp16": FP16,
        "bf16": BF16,
        "report_to": REPORT_TO,

        # DataLoader
        "remove_unused_columns": REMOVE_UNUSED_COLUMNS,
        "dataloader_pin_memory": DATALOADER_PIN_MEMORY,
        "dataloader_num_workers": DATALOADER_NUM_WORKERS,
    }


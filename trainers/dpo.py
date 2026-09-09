from trl import DPOConfig, DPOTrainer

from config import *

from .base import build_common_training_args


def build_training_args():
    return DPOConfig(
        **build_common_training_args(),

        beta=DPO_BETA,
        loss_type=DPO_LOSS_TYPE,
        precompute_ref_log_probs=PRECOMPUTE_REF_LOG_PROBS,
    )


def build_trainer(
    *,
    model,
    ref_model,
    tokenizer,
    train_dataset,
    eval_dataset,
):

    return DPOTrainer(
        model=model,

        ref_model=None,

        args=build_training_args(),

        processing_class=tokenizer,

        train_dataset=train_dataset,

        eval_dataset=eval_dataset,
    )
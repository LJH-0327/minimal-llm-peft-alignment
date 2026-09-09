from trl import SFTTrainer,SFTConfig

from config import *

from .base import build_common_training_args

def build_training_args():
    return SFTConfig(
        **build_common_training_args(),
    )

def build_trainer(
    *,
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    ref_model=None
):

    return SFTTrainer(
        model=model,

        args=build_training_args(),

        processing_class=tokenizer,

        train_dataset=train_dataset,

        eval_dataset=eval_dataset,
    )
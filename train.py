from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
from transformers.trainer_utils import get_last_checkpoint

from datasets import load_from_disk

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)

from config import *

from trainers import build_trainer

import torch

from utils.dataset import (
    split_dataset, 
    print_dataset_statistics
)


def prepare_training(model):
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model

def build_sft_model(base_model):

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=TARGET_MODULES,
    )

    model = get_peft_model(
        base_model,
        lora_config
    )

    return prepare_training(model), None


def build_dpo_model(base_model):

    model = PeftModel.from_pretrained(
        base_model,
        str(SFT_ADAPTER_PATH),
        is_trainable=True
    )

    return prepare_training(model), None

MODEL_BUILDERS = {
    "sft": build_sft_model,
    "dpo": build_dpo_model,
}

def load_models(device):

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.bfloat16
    ).to(device)



    builder = MODEL_BUILDERS[MODE]
    model, ref_model = builder(base_model)

    return tokenizer, model, ref_model



def save_model(trainer, tokenizer):
    trainer.save_model(str(OUTPUT_ADAPTER_PATH))
    tokenizer.save_pretrained(str(OUTPUT_ADAPTER_PATH))



def main():

    print("=" * 50)
    print(f"Model   : {MODEL_NAME}")
    print(f"Dataset : {OUTPUT_DATASET_NAME}")
    print(f"Mode: {MODE}")
    print("=" * 50)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)


    print("Loading model...")
    tokenizer, model, ref_model = load_models(device)


    print("Preparing dataset...")
    dataset = load_from_disk(str(RAW_DATASET_PATH))

    train_dataset = (
        dataset["train"]
        .shuffle(seed=SEED)
        .select(range(6000))
    )

    eval_dataset = (
        dataset["test"]
        .shuffle(seed=SEED)
        .select(range(1000))
    )

    # train_dataset, eval_dataset = split_dataset(dataset)

    print("Train size:", len(train_dataset))
    print("Eval size :", len(eval_dataset))
    if MODE=="sft":
        print_dataset_statistics(train_dataset)


    print("Building trainer...")
    trainer = build_trainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )


    print("Starting training...")
    OUTPUT_ADAPTER_PATH.mkdir(parents=True, exist_ok=True)
    trainer.train(resume_from_checkpoint=get_last_checkpoint(OUTPUT_ADAPTER_PATH))


    print("Saving model...")
    save_model(trainer, tokenizer)

    
    print("Finished")


if __name__ == '__main__':
    main()
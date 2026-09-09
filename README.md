# Minimal LLM PEFT & Alignment Framework

A lightweight and modular framework for experimenting with **parameter-efficient fine-tuning (PEFT)** and **preference alignment** on small language models.

The project is built with **PyTorch, HuggingFace Transformers, PEFT and TRL**, and currently supports:

- LoRA-based Supervised Fine-Tuning (**LoRA-SFT**)
- Direct Preference Optimization (**DPO**)
- Base Model → SFT Adapter → DPO Alignment workflow
- Config-driven switching between models, datasets and training strategies
- Independent management of adapters, checkpoints and evaluation results

The main goal of this project is not to build another large training system, but to provide a **minimal experimental framework** for studying how different models, datasets and post-training methods affect small LLMs.

---

## 1. Motivation

When experimenting with different LLMs and datasets, I found that many parts of the training pipeline were repeatedly rewritten:

- model and tokenizer loading
- LoRA configuration
- dataset selection
- training arguments
- trainer construction
- checkpoint recovery
- adapter saving
- inference and evaluation

The differences became more obvious when moving from SFT to preference-based methods such as DPO.

This project therefore separates **experiment configuration**, **model preparation**, **trainer construction**, **adapter management**, and **evaluation** as much as possible, while keeping a unified training lifecycle.

---

## 2. Framework Overview

```text
                    Experiment Config
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
     Model Config     Dataset Config    Training Mode
          │                │                │
          ▼                │                ▼
 AutoModel/Tokenizer       │         Trainer Registry
          │                │          ┌──────┴──────┐
          │                │          │             │
          │                │         SFT           DPO
          │                │
          └────────────────┼─────────────────────────┘
                           ▼
                    Unified train.py
                           │
                           ▼
                 Adapter / Checkpoint
                           │
                           ▼
                  Inference / Evaluation
```

For the current DPO experiment, the training workflow is:

```text
Qwen2.5-0.5B-Instruct
          │
          ▼
     LoRA-SFT on FiQA
          │
          ▼
      SFT Adapter
          │
          ▼
 DPO on UltraFeedback
          │
          ▼
   Alignment Adapter
```

---

## 3. Main Design

### Config-driven Experiments

Model, dataset, training mode, paths and training hyperparameters are managed in the `config/` directory.

This makes it possible to change experimental combinations without rewriting the main training pipeline.

### Model Builder

Different training strategies may require different model preparation.

For example:

- SFT creates a new LoRA adapter on the base model.
- DPO loads an existing SFT adapter and continues optimization from that state.

Model preparation logic is selected through a model builder mapping.

### Trainer Registry

SFT and DPO use independent trainer modules but expose a similar builder interface.

```python
TRAINER_REGISTRY = {
    "sft": build_sft_trainer,
    "dpo": build_dpo_trainer,
}
```

The main training script therefore does not need to directly instantiate `SFTTrainer` or `DPOTrainer`.

Common training arguments are shared between different trainers, while algorithm-specific parameters remain in their own modules.

### PEFT & Adapter Management

LoRA is used to avoid full-parameter fine-tuning.

Each experiment stores its adapter and relevant training artifacts independently, making it easier to compare different:

```text
Model × Dataset × Training Strategy
```

combinations.

### Dataset & Evaluation

Dataset-specific formatting is handled before the main training stage.

Model-specific tokenization is handled by HuggingFace `AutoTokenizer`, avoiding separate tokenizer implementations for different model families.

Evaluation prompts cover several categories, including:

- general instruction following
- finance
- medicine
- general reasoning

The same prompts are used before and after training to compare changes in model behavior.

---

## 4. Project Structure

```text
.
├── config/                 # experiment configuration
│   ├── dataset.py
│   ├── dpo.py
│   ├── model.py
│   ├── paths.py
│   ├── runtime.py
│   └── training.py
│
├── trainers/
│   ├── base.py             # shared training arguments
│   ├── sft.py              # SFT trainer
│   ├── dpo.py              # DPO trainer
│   └── __init__.py         # trainer registry
│
├── datasets/
│   ├── raw/
│   └── eval_prompts/
│
├── adapters/
│   └── sft/
│
├── inference/
│   ├── chat.py
│   ├── eval_chat.py
│   ├── generate.py
│   └── loader.py
│
├── results/
├── utils/
└── train.py
```

---

## 5. Current Experiments

| Model | Training | Dataset | Purpose |
|---|---|---|---|
| Qwen2.5-0.5B | LoRA-SFT | Dolly-15k | instruction adaptation |
| Qwen2.5-0.5B-Instruct | LoRA-SFT | FiQA | financial-domain adaptation |
| Qwen2.5-0.5B-Instruct | LoRA-SFT | PubMedQA | medical-domain adaptation |
| TinyLlama-1.1B-Chat | LoRA-SFT | FiQA | cross-model comparison |
| Qwen2.5-0.5B-Instruct | LoRA-SFT → DPO | FiQA → UltraFeedback | preference alignment |

---

## 6. Quick Start

### Install dependencies

The main dependencies are:

```bash
pip install torch transformers datasets peft trl
```

> Package versions should be selected according to the local CUDA environment.  
> A version-pinned `requirements.txt` can be added for reproducibility.

### Prepare models and datasets

The current project expects local model and dataset directories configured through `config/paths.py`.

Example structure:

```text
models/
└── Qwen2.5-0.5B-Instruct/

datasets/
└── raw/
    ├── FiQA/
    └── UltraFeedback/
```

Dataset-specific field conversion should be completed before training so that the training stage receives data compatible with SFT or preference training.

### Configure an experiment

Modify the corresponding files in `config/`, for example:

```python
MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODE = "sft"
```

and select the required dataset and training hyperparameters.

### Train

```bash
python train.py
```

If an existing checkpoint is found in the output directory, the training pipeline can resume from the latest checkpoint.

---

## 7. SFT → DPO Workflow

### Stage 1: LoRA-SFT

Set:

```python
MODE = "sft"
```

and select the SFT dataset.

The framework creates a LoRA adapter and saves the trained adapter after training.

### Stage 2: DPO

After obtaining an SFT adapter, change the training mode to:

```python
MODE = "dpo"
```

The framework loads the existing SFT adapter and continues preference optimization using a preference dataset such as UltraFeedback.

---

## 8. Experimental Observations

The experiments produced several interesting observations.

### Instruction SFT can improve base-model stability

For `Qwen2.5-0.5B + Dolly-15k`, LoRA-SFT reduced obvious topic drifting and abnormal text generation, and improved basic instruction-following behavior.

### Domain SFT does not always improve every capability

For some FiQA experiments, domain fine-tuning changed the model's financial response style but also introduced errors in simple calculations and general reasoning.

This suggests that domain adaptation and general capability preservation should be evaluated separately.

### Domain style is not equivalent to factual reliability

In the PubMedQA experiment, the fine-tuned model sometimes generated medical-style content even when the required source information was missing.

This highlights the difference between:

```text
Domain Adaptation
        ≠
Factual Reliability
```

### Preference alignment is not the same as reasoning improvement

After FiQA SFT → UltraFeedback DPO, some answers improved while other simple reasoning tasks became worse.

The current experiments therefore suggest that:

```text
Preference Alignment
        ≠
Factual / Reasoning Accuracy
```

These are separate evaluation dimensions.

---

## 9. Current Limitations

This project is currently an experimental framework rather than a complete LLM training platform.

Current limitations include:

- experiments are mainly based on small language models
- evaluation is still largely based on fixed prompt comparisons
- no complete automated benchmark pipeline yet
- dataset schema validation is not fully implemented
- LoRA hyperparameters have not been systematically ablated
- only SFT and DPO are currently implemented

---

## 10. Roadmap

Planned improvements include:

- [ ] unified SFT / Preference dataset schema validation
- [ ] automated evaluation pipeline
- [ ] quantitative evaluation for reasoning and factuality
- [ ] preference pairwise evaluation / win rate
- [ ] LoRA hyperparameter ablation
- [ ] ORPO support
- [ ] SimPO support
- [ ] further exploration of PPO-style alignment
- [ ] improved component registration and experiment tracking

> ORPO, SimPO and PPO are future work and are **not currently implemented**.

---

## 11. Repository Notes

Large base-model weights and full dataset caches are not recommended to be committed directly to GitHub.

A cleaner repository can keep:

```text
source code
configs
evaluation prompts
small experiment results
adapter configuration
documentation
```

while excluding large files such as:

```text
*.safetensors
models/
datasets/raw/
checkpoint-*/
*.arrow
```

These files can be added to `.gitignore`.

---

## 12. About This Project

This project was developed as a personal exploration of:

- parameter-efficient LLM fine-tuning
- preference alignment
- modular training framework design
- model behavior evaluation
- the relationship between training objectives and model reliability

One of the main lessons from the experiments is that post-training should not be viewed simply as making a model “better”.

Different combinations of **model, data, optimization objective and evaluation method** can lead to very different capability changes.

Understanding these interactions is one of the directions I hope to explore further.

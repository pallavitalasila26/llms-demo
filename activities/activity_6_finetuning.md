# Activity 6: Fine-tune a small model with LoRA

**Objective:** Fine-tune a small language model on a custom instruction dataset using LoRA, then compare its behaviour against the original base model.

**Duration:** 60-90 minutes

---

## Overview

In this activity you will:

1. Load a small base model (`Qwen/Qwen2.5-0.5B`) and observe its default behaviour
2. Build a tiny instruction dataset (10-20 examples) on a topic of your choice
3. Fine-tune the model with LoRA using `SFTTrainer` from the `trl` library
4. Run the fine-tuned adapter alongside the base model and compare responses

This is the smallest realistic fine-tuning experiment you can run on a single consumer GPU. The same pattern scales directly to larger models (1.5B, 3B, 7B …) and larger datasets.

---

## Setup

### 1. Install dependencies

```bash
pip install peft trl
```

`peft` provides LoRA. `trl` provides `SFTTrainer`, a wrapper around the HuggingFace `Trainer` class that handles the SFT-specific details.

### 2. Create your activity script

Create a file `activities/my_finetuning.py`. All code snippets below go into this file.

```python
"""Activity 6: LoRA fine-tuning experiment."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
```

### 3. Verify GPU access

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
```

---

## Part 1: Baseline - observe the base model

Before any fine-tuning, establish what the base model does *without* instruction tuning.

### Step 1: Load the base model

```python
MODEL_ID = "Qwen/Qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16)
model = model.to(device)
model.eval()
```

### Step 2: Write a helper to generate text

```python
def generate(model, tokenizer, prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
```

### Step 3: Run some test prompts

Pick a topic for your fine-tuning experiment - something the base model handles
poorly. Good options:

- **Consistent persona** - always respond as a particular character
- **Structured output** - always reply in a specific format (JSON, bullet points)
- **Domain tone** - always explain things using analogies from a chosen field
- **Short answers** - always answer in exactly one sentence

Run your chosen prompts through the base model and note the responses. These are
your baseline - you will compare against them after fine-tuning.

```python
test_prompts = [
    "What is machine learning?",
    "Explain neural networks.",
    "What is overfitting?",
]

print("=== BASE MODEL RESPONSES ===")
for prompt in test_prompts:
    response = generate(model, tokenizer, prompt)
    print(f"\nPrompt: {prompt}")
    print(f"Response: {response}")
```

**Questions to consider:**
- Does the base model follow the prompt format you want, or does it treat it as text to continue?
- How consistent is the tone and structure across different prompts?

---

## Part 2: Build an instruction dataset

### Step 1: Choose a format and topic

SFT training data is a list of `(instruction, response)` pairs. Pick a topic and
a specific style you want to teach. A few ideas:

- *"Always explain concepts using cooking analogies"*
- *"Always answer in exactly one sentence, no more"*
- *"Always structure answers as: Definition | Example | Limitation"*

### Step 2: Write 10-20 examples

Keep responses short and consistent - the style you write is the style the model
will learn to imitate, so pick one pattern and stick to it across *all* examples.

```python
raw_examples = [
    {
        "instruction": "What is machine learning?",
        "response": "Machine learning is like teaching a recipe to someone who ...",
    },
    {
        "instruction": "Explain neural networks.",
        "response": "A neural network is like a ...",
    },
    # Add more examples here - aim for 15-20
]
```

**Tips for effective examples:**
- Make the response style *unmistakably* different from the base model's default
- Be consistent - if example 1 uses a cooking analogy, all examples should too
- Keep responses under 100 words so training stays fast
- Cover a variety of questions, not just paraphrases of the same question

### Step 3: Format as ChatML and create a Dataset

`SFTTrainer` needs the data in a single `text` field containing the full formatted
conversation, including special tokens. Qwen uses ChatML format:

```python
def format_chatml(example):
    text = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{example['instruction']}<|im_end|>\n"
        f"<|im_start|>assistant\n{example['response']}<|im_end|>\n"
    )
    return {"text": text}

dataset = Dataset.from_list(raw_examples)
dataset = dataset.map(format_chatml)
print(dataset[0]["text"])  # inspect one example to confirm formatting
```

---

## Part 3: Configure and attach LoRA

### Step 1: Choose LoRA hyperparameters

```python
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,             # rank - number of trainable dimensions per weight matrix
    lora_alpha=16,   # scaling factor (alpha/r controls effective learning rate)
    target_modules=["q_proj", "v_proj"],  # which weight matrices to adapt
    lora_dropout=0.05,
    bias="none",
)
```

**What these control:**
- `r` - higher rank = more expressive adapter, more parameters, slower training. `r=8` is a safe default for small datasets.
- `target_modules` - LoRA is typically applied to the attention `q` and `v` projection matrices. You can also add `k_proj`, `o_proj`, and the MLP layers for more capacity.
- `lora_alpha` - scalar that multiplies the adapter output. The effective learning rate for the adapter scales with `alpha / r`.

### Step 2: Attach the adapter

```python
model.train()  # switch back to training mode
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()
```

`print_trainable_parameters()` shows you how many parameters are trainable vs frozen. For a 0.5B model with `r=8` on q/v projections you should see roughly **0.5-1% trainable**.

**Question to consider:**
- How does the trainable parameter count change if you add `k_proj` and `o_proj` to `target_modules`?

---

## Part 4: Train

```python
training_args = SFTConfig(
    output_dir="./lora_output",
    num_train_epochs=5,        # more epochs = more repetition of your examples
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,  # effective batch size = 2 * 4 = 8
    learning_rate=2e-4,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=5,
    save_strategy="no",        # skip checkpoints to keep things fast
    fp16=True,
    report_to="none",          # disable wandb/tensorboard
    max_seq_length=256,
)

trainer = SFTTrainer(
    model=peft_model,
    train_dataset=dataset,
    args=training_args,
)

trainer.train()
```

Training a tiny dataset for 5 epochs on a single GPU takes **1-3 minutes**. Watch
the loss in the logs - it should decrease. If it barely moves, try increasing
`num_train_epochs` or `learning_rate`.

---

## Part 5: Compare base vs fine-tuned

Set the trained model to eval mode and test with the same prompts you used in Part 1:

```python
peft_model.eval()

print("=== FINE-TUNED MODEL RESPONSES ===")
for prompt in test_prompts:
    # Apply the chat template so the model sees the format it was trained on
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    response = generate(peft_model, tokenizer, formatted)
    print(f"\nPrompt: {prompt}")
    print(f"Response: {response}")
```

**Questions to consider:**
- Does the fine-tuned model consistently apply the style you taught it?
- Does it still answer correctly, or did it sacrifice accuracy for style?
- Try a prompt that *wasn't* in your training set - does the style generalise?

---

## Part 6: Examine what changed

### The adapter weights

LoRA stores only the lightweight adapter - not a full copy of the model.

```python
# Save only the LoRA adapter (small - a few MB)
peft_model.save_pretrained("./my_lora_adapter")

import os
adapter_size = sum(
    os.path.getsize(os.path.join("./my_lora_adapter", f))
    for f in os.listdir("./my_lora_adapter")
) / 1024 / 1024
print(f"Adapter size: {adapter_size:.1f} MB")
```

Compare this to the base model size (~1 GB). The adapter captures the new style
in a tiny fraction of the original parameter space.

### Load the adapter separately

```python
from peft import PeftModel

# Load the original base model fresh
base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to(device)

# Load the adapter on top
finetuned_model = PeftModel.from_pretrained(base_model, "./my_lora_adapter")
finetuned_model.eval()

# Confirm it produces the same outputs as peft_model above
response = generate(finetuned_model, tokenizer, formatted)
print(response)
```

This demonstrates the deployment pattern: ship the base model once, swap adapters
to change behaviour without touching the 1 GB base checkpoint.

---

## Extension challenges

If you finish early, try one or more of these:

### A. Increase rank and measure the tradeoff

Re-run training with `r=16` and `r=32`. For each:
- Check `print_trainable_parameters()` - how many more parameters?
- Does the fine-tuned behaviour improve, stay the same, or overfit?

### B. Train longer and observe overfitting

Set `num_train_epochs=20` and run again. Does the model start memorising your
training examples word-for-word instead of generalising the style?

### C. Expand target modules

Add `k_proj`, `o_proj`, and `gate_proj` to `target_modules`. Does the larger
adapter produce a stronger or more flexible style shift with the same number of
training examples?

### D. Merge and export

Merge the LoRA adapter permanently into the base model weights and save a
standalone checkpoint:

```python
merged = peft_model.merge_and_unload()
merged.save_pretrained("./merged_model")
tokenizer.save_pretrained("./merged_model")
print("Saved merged model - no PEFT dependency needed to load it")
```

Load and test `./merged_model` with plain `AutoModelForCausalLM` (no `PeftModel`)
to confirm it works identically to the adapter version.

---

## Summary

| Step | What you did | Why it matters |
|------|-------------|----------------|
| Baseline | Ran the base model on test prompts | Establishes ground truth before any training |
| Dataset | Wrote 10-20 instruction/response pairs | Data quality and consistency drives LoRA results |
| LoRA config | Set rank, alpha, target modules | These hyperparameters control adapter capacity |
| Training | Ran `SFTTrainer` for a few epochs | Loss decreasing confirms the adapter is learning |
| Comparison | Side-by-side base vs fine-tuned | Makes the behavioural change concrete and measurable |
| Adapter size | Measured saved adapter vs base | Demonstrates the efficiency case for LoRA |

The same workflow - dataset, LoRA config, SFTTrainer, compare - is what
production fine-tuning pipelines use, just with larger models and more data.

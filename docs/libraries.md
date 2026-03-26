# Libraries

## HuggingFace Transformers

Direct model loading and inference in Python without an external server.

**Links:**
- [Documentation](https://huggingface.co/docs/transformers)
- [Model Hub](https://huggingface.co/models)
- [GitHub](https://github.com/huggingface/transformers)

### Key classes

| Class | Purpose |
|-------|---------|
| `AutoModelForCausalLM` | Loads a causal language model (next-token prediction) |
| `AutoTokenizer` | Loads the tokenizer for a model |
| `.from_pretrained()` | Downloads and caches model/tokenizer from HuggingFace Hub |
| `.apply_chat_template()` | Formats a list of messages into the model's expected format |
| `.generate()` | Performs autoregressive token generation |

### Environment variables

| Variable | Purpose |
|----------|----------|
| `HF_HOME` | Base directory for HuggingFace files (default `~/.cache/huggingface`) |
| `HF_TOKEN` | HuggingFace API token (required for gated models) |

### Example usage

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

# Load model and tokenizer
model_name = 'google/gemma-2-2b-it'
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='auto',
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create a chat prompt
messages = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': 'Hello!'},
]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

# Generate response
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.7,
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```


---

## LangChain

High-level framework for building LLM applications with chains, agents, and RAG pipelines.

**Links:**
- [Documentation](https://python.langchain.com/docs)
- [Ollama integration](https://python.langchain.com/docs/integrations/chat/ollama)
- [Other chat integrations](https://docs.langchain.com/oss/python/integrations/chat)

### Key classes

| Class | Purpose |
|-------|----------|
| `ChatOllama` | LLM client that connects to Ollama |
| `SystemMessage` | Sets the model's behavior/persona |
| `HumanMessage` | A user message |
| `AIMessage` | A model response |

### Example usage

```python
from langchain_ollama import ChatOllama

# Create a client
llm = ChatOllama(
    model='qwen2.5:3b',
    base_url='http://localhost:11434',
    temperature=0.7,
)

# Single request
response = llm.invoke('Tell me a joke')
print(response.content)

# Streaming
for chunk in llm.stream('Count to 10'):
    print(chunk.content, end='', flush=True)
```

### Message types

LangChain uses typed message objects for structured conversations:

```python
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)

messages = [
    SystemMessage(content='You are a helpful assistant.'),
    HumanMessage(content='Hello!'),
    AIMessage(content='Hi there! How can I help?'),
    HumanMessage(content='What is 2+2?'),
]

response = llm.invoke(messages)
```

---

## Gradio

Build web UIs for LLM chat apps with minimal code.

**Links:**
- [Documentation](https://www.gradio.app/docs)
- [GitHub](https://github.com/gradio-app/gradio)
- [ChatInterface guide](https://www.gradio.app/guides/creating-a-chatbot-fast)

### ChatInterface parameters

| Parameter | Purpose |
|-----------|----------|
| `fn` | Function that takes `(message, history)` and returns a response |
| `type` | `'messages'` = structured history with roles, `'tuples'` = simple pairs |
| `title` | Display title |
| `description` | Optional subtitle |
| `examples` | List of example prompts as buttons |
| `additional_inputs` | Extra UI components (textboxes, sliders, etc.) |

### Example usage

```python
import gradio as gr

def chat(message, history):
    # Your LLM call here
    return f'Echo: {message}'

demo = gr.ChatInterface(
    fn=chat,
    title='My Chatbot',
    type='messages',  # Enable structured history
)

demo.launch()
```

---

## PEFT

Parameter-efficient fine-tuning methods that adapt a pre-trained model by training only a small number of additional parameters, leaving the original weights frozen.

**Links:**
- [Documentation](https://huggingface.co/docs/peft)
- [GitHub](https://github.com/huggingface/peft)

### Key classes

| Class | Purpose |
|-------|---------|
| `LoraConfig` | Defines LoRA hyperparameters: rank, alpha, target modules |
| `get_peft_model()` | Wraps a base model with a PEFT adapter |
| `PeftModel.from_pretrained()` | Loads a saved adapter on top of a base model |
| `.print_trainable_parameters()` | Shows how many parameters are trainable vs. frozen |
| `.save_pretrained()` | Saves only the adapter weights (a few MB, not the full model) |
| `.merge_and_unload()` | Merges the adapter into the base weights and returns a plain model |

### Example usage

```python
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# Load the base model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.float16)

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                              # rank
    lora_alpha=16,                    # scaling factor
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)

# Attach the adapter - base model weights are frozen automatically
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()
# trainable params: 786,432 || all params: 494,476,288 || trainable%: 0.1590

# Save only the small adapter
peft_model.save_pretrained("./my_adapter")

# Later: load adapter on top of the base model
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.float16)
loaded = PeftModel.from_pretrained(base, "./my_adapter")
```

---

## TRL

Transformer Reinforcement Learning - a library building on top of HuggingFace Transformers and PEFT to provide trainers for SFT, DPO, PPO, and other alignment methods.

**Links:**
- [Documentation](https://huggingface.co/docs/trl)
- [GitHub](https://github.com/huggingface/trl)

### Key classes

| Class | Purpose |
|-------|---------|
| `SFTConfig` | Training hyperparameters for supervised fine-tuning |
| `SFTTrainer` | Trainer subclass for SFT; handles chat template formatting, packing, and PEFT integration |

### Example usage

```python
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

# Dataset must have a `text` field with fully-formatted conversations
examples = [
    {"text": "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi!<|im_end|>\n"},
    # ... more examples
]
dataset = Dataset.from_list(examples)

training_args = SFTConfig(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    max_seq_length=256,
    report_to="none",
)

trainer = SFTTrainer(
    model=peft_model,        # a PEFT-wrapped model from get_peft_model()
    train_dataset=dataset,
    args=training_args,
)

trainer.train()
```

---

## HuggingFace evaluate

Unified API for computing text generation metrics (ROUGE, BLEU, BERTScore, and many others) with a consistent interface across all metric families.

**Links:**
- [Documentation](https://huggingface.co/docs/evaluate)
- [GitHub](https://github.com/huggingface/evaluate)
- [Available metrics](https://huggingface.co/evaluate-metric)

### Key functions

| Function | Purpose |
|----------|---------|
| `evaluate.load(name)` | Load a metric by name (downloads and caches on first use) |
| `.compute(predictions, references)` | Compute the metric for a batch of predictions |

### Example usage

```python
import evaluate

# ROUGE (summarization / text generation)
rouge = evaluate.load("rouge")
result = rouge.compute(
    predictions=["The cat sat on the mat."],
    references=["A cat was sitting on a mat."],
    use_stemmer=True,
)
# result: {'rouge1': 0.727, 'rouge2': 0.444, 'rougeL': 0.727, 'rougeLsum': 0.727}

# BLEU (translation / n-gram precision)
bleu = evaluate.load("bleu")
result = bleu.compute(
    predictions=[["the", "cat", "sat", "on", "mat"]],     # tokenised
    references=[[["a", "cat", "was", "sitting", "on", "mat"]]],
)
# result: {'bleu': 0.134, ...}

# BERTScore (semantic similarity via contextual embeddings)
bertscore = evaluate.load("bertscore")
result = bertscore.compute(
    predictions=["The cat sat on the mat."],
    references=["A cat was sitting on a mat."],
    lang="en",
    model_type="distilbert-base-uncased",
)
# result: {'precision': [0.96], 'recall': [0.95], 'f1': [0.95], ...}
```

---

## bert-score

Underlying library for BERTScore — contextual embedding similarity metric based on BERT-family models. Used automatically via the `evaluate` `"bertscore"` metric but can also be called directly.

**Links:**
- [Paper](https://arxiv.org/abs/1904.09675)
- [GitHub](https://github.com/Tiiiger/bert_score)

### When to use BERTScore

BERTScore complements ROUGE and BLEU by measuring *semantic similarity* rather than surface-form overlap. It excels at:
- Detecting paraphrases that share meaning but few exact words
- Penalising responses that change key entities (e.g. dates, names) — to a degree
- Evaluating open-ended generation where many valid wordings exist

It does **not** reliably detect subtle factual errors (e.g. swapping "1887" for "1899").

```python
# Direct usage (also available via evaluate.load("bertscore"))
from bert_score import score

P, R, F1 = score(
    cands=["Paris's iconic iron tower was built for a World's Fair."],
    refs=["The Eiffel Tower was built for the 1889 World's Fair in Paris."],
    lang="en",
    model_type="distilbert-base-uncased",
    verbose=True,
)
print(f"BERTScore F1: {F1.mean():.3f}")  # ~0.91 for this paraphrase
```

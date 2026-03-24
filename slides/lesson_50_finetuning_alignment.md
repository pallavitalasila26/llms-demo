---
marp: true
theme: default
paginate: true
style: |
  section {
    background-color: #1a1a2e;
    color: #e0e0e0;
  }
  section h1, section h2, section h3 {
    color: #e2b55a;
  }
  section a {
    color: #93c5fd;
  }
  section strong {
    color: #f0f0f0;
  }
  section table th {
    background-color: #2a2a4a;
    color: #e2b55a;
  }
  section table td {
    background-color: #1e1e36;
    color: #e0e0e0;
  }
  section code {
    background-color: #2a2a4a;
    color: #e0e0e0;
  }
  section pre {
    background-color: #12122a;
  }
  section::after {
    color: #888;
  }
---


# Lesson 50: Fine-tuning, RLHF, and model alignment

**Adapting models and shaping behavior**

---

## Recap: LangChain advanced features
- **Memory** - giving LLMs conversation history
- **Document pipelines** - load → split → embed → store → retrieve
- **Agents** - LLM as a dynamic decision-maker using the ReAct pattern

**Today:** How models are adapted and trained to be more useful and safer

---

## Today's outline

1. **Fine-tuning** - updating model weights for a specific task or style
2. **The alignment problem** - why capable models can be harmful
3. **Reinforcement learning from human feedback (RLHF)** - how we teach models human preferences
4. **Practical guide** - when to fine-tune, what tools to use

After the slides: **Demo 8** compares a base model and instruction-tuned model side by side.

---

# Fine-tuning

Updating model weights for a specific task or style

- How fine-tuning works
- Pre-training vs fine-tuning
- Base model behavior
- Why fine-tune?
- Full fine-tuning and LoRA / QLoRA
- Dataset formats

---

## How fine-tuning works

**Step 1 - Load a pre-trained checkpoint.** Start from a model that already knows language - don't train from scratch.

**Step 2 - Prepare your dataset.** Format examples as `(input, target)` pairs matched to your task (e.g., instruction/response, document/summary).

**Step 3 - Run gradient descent.** The model generates predictions, a loss function measures the error against the target, and gradients update the weights.

**Step 4 - Monitor carefully.** Watch for overfitting on small datasets and catastrophic forgetting of general capabilities.

**Step 5 - Save and deploy.**

---

## Pre-training vs fine-tuning

| | Pre-training | Fine-tuning |
|---|---|---|
| **Goal** | Learn language and world knowledge | Adapt to a task, style, or domain |
| **Data** | Trillions of tokens from the web | Thousands to millions of curated examples |
| **Compute** | Months on thousands of GPUs | Hours to days on a single GPU |
| **What changes** | All weights, from scratch | All or a subset of weights, from a checkpoint |
| **Who does it** | Model labs (OpenAI, Meta, Google) | Researchers, companies, practitioners |

**Key insight:** Fine-tuning starts from a pre-trained checkpoint - you borrow the model's knowledge and redirect it.

---

## Base model behavior

A base model is a **text completion engine** - it predicts the next token.

**Given this prompt:**
```
User: How do I bake a chocolate cake?
```

**A base model might respond with:**
```
User: How do I bake a chocolate cake?
Assistant: How do I bake a lemon tart?
User: Can someone help with recipes?
```

It the pattern of a Q&A forum - it doesn't "know" it's supposed to answer.

**Base models are not instruction followers.** They require fine-tuning to become useful assistants.

---

## Why fine-tune?

Four main reasons to fine-tune rather than just prompt:

1. **Task specialization** - a model fine-tuned on legal documents outperforms a general model on legal tasks
2. **Style consistency** - lock in a specific tone or format (e.g., always output JSON)
3. **Domain knowledge** - inject terminology, conventions, and facts that weren't in pre-training data
4. **Efficiency** - a fine-tuned 3B model can match a 70B model on a narrow task, at a fraction of the cost

**Fine-tuning vs RAG:**
- RAG is better for recent facts and large knowledge bases
- Fine-tuning is better for behavior, style, format, and skills

---

## Full fine-tuning

**Full fine-tuning** updates every parameter in the model.

**Pros:**
- Maximum expressiveness - the model can change in any direction

**Cons:**
- Requires storing a full copy of the model for each task
- Catastrophic forgetting - original knowledge can degrade
- High GPU memory requirement (store weights + gradients + optimizer states)

---

## Full fine-tuning: VRAM requirement

| Model size | VRAM for full fine-tune (fp16) |
|---|---|
| 7B | ~80 GB |
| 13B | ~160 GB |
| 70B | ~800 GB |

---

## Instruction tuning

**Instruction tuning** is full fine-tuning on a dataset of `(instruction, output)` pairs.

**Training example:**

```text
{
  "instruction": "Summarize the following article in one sentence.",
  "input": "Scientists at MIT have developed a new battery...",
  "output": "MIT researchers created a battery technology that charges 10x faster using a new electrode material."
}
```

Famous instruction-tuning datasets: **Alpaca** (52k examples), **FLAN** (1,800+ tasks), **OpenHermes** (1M+ examples)

**Key insight:** The model already knows how to summarize - instruction tuning teaches it *when* to do so.

---

## LoRA: How training works

**Step 1 - Freeze the base model.** All original weights W are locked — no gradients flow through them.

**Step 2 - Inject adapter pairs.** For each target layer (usually the attention projection matrices), add two small trainable matrices A and B initialized so B·A = 0 at the start of training.

**Step 3 - Train only the adapters.** Forward pass uses `W·x + (B·A)·x`. Gradients update only A and B.

**Step 4 - Merge or keep separate.** At inference you can either:
- **Merge:** fold B·A into W (ΔW = B·A), zero extra cost at runtime
- **Keep separate:** swap adapters in/out to serve multiple fine-tunes from one base model

---

## LoRA: Low-Rank Adaptation

**LoRA** (Hu et al., 2021) fine-tunes a small number of new parameters while keeping the original weights frozen.

Instead of updating the full weight matrix **W**, LoRA learns a low-rank decomposition **ΔW = B × A**:

```text
Original:  output = W · x
LoRA:      output = W · x  +  (B · A) · x
                    frozen     trainable
```

**Why two matrices and not one?** With r=8 on a 4096×4096 layer: a full ΔW costs **16.7M** params, but B·A costs only **65K** (~256× smaller).

The bottleneck rank `r` forces adaptation through a low-dimensional subspace - the same idea as PCA/SVD.

---

## LoRA: VRAM requirement

| Method | Trainable params | VRAM savings |
|---|---|---|
| Full fine-tune | 100% | - |
| LoRA (r=8) | ~0.1% | 3–5× |
| LoRA (r=64) | ~0.5% | 2–3× |

| Model size | LoRA VRAM | Consumer GPU |
|---|---|---|
| 3B | ~4 GB | RTX 3060 (12 GB), RTX 4060 (8 GB) |
| 7B | ~8 GB | RTX 3080 (10 GB), RTX 4070 (12 GB) |
| 13B | ~14 GB | RTX 3090 / 4090 (24 GB) |

**Result:** Near full fine-tune quality - a 7B LoRA fine-tune fits on a mid-range gaming GPU.

---

## QLoRA: Quantization + LoRA

**QLoRA** (Dettmers et al., 2023) combines LoRA with 4-bit quantization of the base model.

**NF4 (Normal Float 4):** A quantization format optimized for normally-distributed weights - better quality than standard int4.

**Double quantization:** Quantize the quantization constants themselves, saving an additional ~0.4 bits/param.

| Model | Full fine-tune VRAM | QLoRA VRAM |
|---|---|---|
| 7B | ~80 GB | ~5 GB |
| 13B | ~160 GB | ~9 GB |
| 65B | ~780 GB | ~40 GB |

---

## Dataset formats

Fine-tuning requires structured training data. Three common formats:

- **Alpaca (SFT - Supervised Fine-Tuning)**
- **ChatML (SFT - conversational)**
- **DPO pair (preference learning)**

---

**Alpaca (SFT - Supervised Fine-Tuning):**
```text
{
  "instruction": "Translate the following to French.",
  "input": "The weather is nice today.",
  "output": "Le temps est agréable aujourd'hui."
}
```

---

**ChatML (SFT - conversational):**
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
The capital of France is Paris.<|im_end|>
```

---

**DPO pair (preference learning):**
```text
{
  "prompt": "Explain quantum entanglement simply.",
  "chosen": "Imagine two coins that always land opposite...",
  "rejected": "Quantum entanglement involves the superposition of..."
}
```

---

# The alignment problem

Why capable models can still harmful

---

## What is alignment?

A model is **aligned** when its behavior matches human values and intentions.

The three goals - **HHH:**

| Goal | Meaning | Example failure |
|---|---|---|
| **Helpful** | Answers questions accurately, follows instructions | Refuses reasonable requests, hedges excessively |
| **Harmless** | Avoids content that could cause harm | Provides instructions for dangerous activities |
| **Honest** | Doesn't deceive or confabulate | States false information with high confidence |

**The tension:** These goals can conflict! (see HAL 9000 and '2001: A Space Odyssey').

---

# Reinforcement learning from human feedback

Teaching models human preferences

---

## The RLHF pipeline

Three stages to go from a base model to an aligned assistant:

```
Stage 1                  Stage 2                    Stage 3
─────────────────        ──────────────────────     ──────────────────────
Pre-trained model   →    Reward model          →    PPO training
+ SFT on                 trained on human           aligns policy model
  instruction data       preference pairs           to reward model
─────────────────        ──────────────────────     ──────────────────────
SFT model                Reward model               Aligned model
(instruction-             (scores responses)         (InstructGPT, Claude,
 following)                                           Llama-3-instruct)
```

This is how **InstructGPT** (the model behind early ChatGPT) was built - described in [Ouyang et al. (2022) "Training language models to follow instructions with human feedback"](https://arxiv.org/abs/2203.02155).

---

## Stage 1: Supervised fine-tuning (SFT)

**Goal:** Give the base model a foundation in instruction-following before RL training.

**Process:**
1. Collect a set of prompts, have human labelers write ideal responses
2. Fine-tune the base model on these `(prompt, response)` pairs

**Why this is necessary:**
- RL training is unstable without a good starting point
- SFT teaches the model the *format* of helpful responses
- Reduces the search space for the reward-guided optimization

**Result:** A model that follows instructions, no human preferences yet.

---

## Stage 2: Reward model

**Goal:** Learn a function that scores responses by how much humans prefer them.

**Process:**
1. Sample multiple responses to the same prompt from the SFT model
2. Have human raters rank responses (which is better?)
3. Train a **reward model** (RM) to predict the human ranking

**The preference pair:**
```
Prompt:   "Explain black holes to a 10-year-old."
Chosen:   "Imagine a giant vacuum cleaner in space..."  → RM score: 0.92
Rejected: "A black hole is a region of spacetime..."    → RM score: 0.31
```

The reward model is itself an LLM with a scalar output head - it outputs a single number representing "how good is this response."

---

## Stage 3: PPO policy optimization

**Goal:** Train the policy (the LLM that talks to users) to maximize the reward model's score.

- Generate responses, score with the reward model, update weights to increase the score
- A **KL divergence penalty** keeps the policy close to the SFT model, preventing "reward hacking" — finding responses that fool the reward model without being genuinely good

```
Objective = E[RM(response)] - β × KL(policy || SFT_model)
             ↑ maximize        ↑ don't drift too far
```

---

## DPO: Direct Preference Optimization

**DPO** (Rafailov et al., 2023) achieves similar alignment to RLHF without a separate reward model or PPO.

**Key insight:** The optimal policy under RLHF has a closed form - you can derive it from preference data directly.

| | RLHF (PPO) | DPO |
|---|---|---|
| **Training stages** | 3 (SFT + RM + PPO) | 2 (SFT + DPO) |
| **Reward model** | Required | Not needed |
| **Stability** | Finicky, needs tuning | More stable |
| **Memory cost** | High (policy + RM + ref) | Moderate (policy + ref) |
| **Quality** | Strong | Often comparable |

**Most modern open-source fine-tunes** (Mistral-Instruct, Llama-3-Instruct, Qwen2.5-Instruct) use DPO or a variant.

---

## Constitutional AI

**Constitutional AI** (Anthropic, 2022) extends RLHF with an AI-generated feedback loop.

**The idea:**

1. Give the model a **constitution** - a set of principles (e.g., "don't assist with illegal activities")
2. Have the model **critique its own responses** against the constitution
3. Have the model **revise** its responses based on the critique
4. Use the revised responses as SFT data and preference pairs

**RLAIF (RL from AI Feedback):** The reward model is trained on AI-labeled preferences, not just human-labeled ones - dramatically scaling the amount of feedback signal available.

This is how **Claude** (Anthropic's model) is trained to be helpful, harmless, and honest.

---

# Practical guide

When to fine-tune, and what tools to use

---

## Fine-tune or RAG?

| Scenario | Recommendation | Reason |
|---|---|---|
| **Recent or frequently updated facts** | RAG | Fine-tuning can't keep up with changing data |
| **Large private knowledge base** | RAG | More scalable than encoding into weights |
| **Consistent output format/style** | Fine-tune | Format is a learned behavior, not a retrieval problem |
| **Domain-specific vocabulary and conventions** | Fine-tune | Terminology shapes how the model reasons |
| **Reducing hallucination on known content** | RAG | Grounding with retrieved text is more reliable |
| **Faster inference (no retrieval step)** | Fine-tune | Weights hold knowledge; no retrieval at runtime |
| **Both facts and behavior** | Both | Fine-tune for behavior; RAG for knowledge |

---

## Tools landscape

| Tool | Purpose | Notes |
|---|---|---|
| **Unsloth** | Fast LoRA/QLoRA fine-tuning | 2–5× faster than HF PEFT; great for single-GPU |
| **PEFT** | Hugging Face adapter library | LoRA, prefix tuning, prompt tuning; integrates with Trainer |
| **TRL** | Transformer Reinforcement Learning | SFTTrainer, DPOTrainer, PPOTrainer; official HF RL library |
| **axolotl** | Config-driven fine-tuning | YAML config files; supports many dataset formats |
| **LlamaFactory** | UI + code for fine-tuning | Web UI for dataset prep and training runs |

**Starting point:** Unsloth + TRL SFTTrainer is the fastest path to a working fine-tuned model on a single GPU.

**Models:** Hugging Face Hub hosts thousands of base checkpoints and LoRA adapters - see [huggingface.co/models](https://huggingface.co/models)

---

## What we covered today

1. **Fine-tuning** - update model weights to specialize; full fine-tune, instruction tuning, LoRA, QLoRA
2. **Dataset formats** - Alpaca, ChatML, DPO pairs - what training data actually looks like
3. **Alignment problem** - base models aren't assistants; HHH goals and their tensions
4. **RLHF** - three-stage pipeline: SFT -> reward model -> PPO to teach human preferences
5. **DPO** - modern alternative to PPO that skips the reward model
6. **Practical guide** - when to fine-tune vs RAG; Unsloth, PEFT, TRL, axolotl

---

## Additional resources

**Papers:**
- [LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314)
- [Training language models to follow instructions with human feedback (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155)
- [Direct Preference Optimization (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290)
- [Constitutional AI (Bai et al., 2022)](https://arxiv.org/abs/2212.08073)

**Hugging Face:**
- [PEFT documentation](https://huggingface.co/docs/peft)
- [TRL documentation](https://huggingface.co/docs/trl)
- [Unsloth](https://github.com/unslothai/unsloth)

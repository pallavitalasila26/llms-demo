# Slides

This section contains lecture slides covering LLM fundamentals, deployment, and prompting techniques.

> **Note:** Slide files are Markdown documents formatted for [Marp](https://marp.app/). You can view them in the [GitHub repository](https://github.com/gperdrizet/llms-demo/tree/main/slides) or render them as slides in VS Code:
>
> 1. Open a slide file (e.g., `slides/lesson_44_state_of_the_art.md`)
> 2. Click the **Marp icon** in the top-right corner of the editor, or
> 3. Press `Ctrl+Shift+P` and run **Marp: Open Preview**
>
> The Marp extension is pre-installed in the dev container.

## Lesson 44: State of the art in generative AI

Overview of the current LLM landscape, model architectures, and foundational concepts.

**Topics covered:**
- Decoder-only transformers
- Tokenization and autoregressive generation
- Training approaches
- Current state-of-the-art models

**Location:** `slides/lesson_44_state_of_the_art.md`

## Lesson 45: LLM deployment

Practical approaches to deploying and serving LLMs in production and development environments.

**Topics covered:**
- Inference servers (Ollama, llama.cpp)
- Model quantization and optimization
- CPU/GPU memory management
- API interfaces

**Location:** `slides/lesson_45_llm_deployment.md`

## Lesson 46: Prompting fundamentals

Introduction to prompt engineering and basic techniques for effective LLM interaction.

**Topics covered:**
- Prompt structure and formatting
- Zero-shot and few-shot learning
- Chain-of-thought reasoning
- System prompts and role definition

**Location:** `slides/lesson_46_prompting_fundamentals.md`

## Lesson 47: Advanced prompting

Advanced techniques for complex tasks and improving model outputs.

**Topics covered:**
- Self-consistency and multiple reasoning paths
- Reflection and critique
- Decomposition and step-by-step reasoning
- Prompt optimization strategies

**Location:** `slides/lesson_47_advanced_prompting.md`

## Lesson 48: LangChain basics

Introduction to building structured LLM applications with LangChain.

**Topics covered:**
- Chat models and LLM wrappers
- Chat prompt templates and structured prompts
- Output parsers (string, JSON, Pydantic)
- Basic chains and composition with LCEL

**Location:** `slides/lesson_48_langchain_basics.md`

## Lesson 49: LangChain advanced features

Advanced LangChain patterns: conversational memory, the RAG pipeline, and agents.

**Topics covered:**
- Conversational memory strategies (`ConversationBufferMemory`, `ConversationSummaryMemory`, etc.)
- Document loading, splitting, and embedding
- Vector stores and retrievers (RAG pipeline)
- Agents and the ReAct pattern
- Tools and LangChain agent components

**Location:** `slides/lesson_49_langchain_advanced.md`

## Lesson 50: Fine-tuning, RLHF, and model alignment

How pre-trained base models are adapted into instruction-following assistants through supervised fine-tuning and reinforcement learning from human feedback.

**Topics covered:**
- Base model behaviour vs. instruction-tuned behaviour
- Supervised fine-tuning (SFT) on instruction/response pairs
- LoRA and QLoRA: parameter-efficient fine-tuning
- RLHF pipeline: reward models, PPO, and preference data
- Direct Preference Optimization (DPO) as an RLHF alternative
- Practical VRAM requirements for consumer and server GPUs

**Location:** `slides/lesson_50_finetuning_alignment.md`

## Lesson 51: Benchmarking and evaluating LLMs

How to measure LLM output quality using automated text metrics, standardised benchmarks, and LLM-as-judge scoring.

**Topics covered:**
- Why evaluation is hard (no ground truth, multiple quality dimensions)
- Automated text metrics: ROUGE, BLEU, and BERTScore
- Standard benchmarks: MMLU, HellaSwag, GSM8K, HumanEval, TruthfulQA
- Leaderboards, data contamination, and safety evaluation
- LLM-as-judge: rubric scoring, pairwise comparison, and failure modes
- Evaluation frameworks: HuggingFace `evaluate`, lm-evaluation-harness, RAGAS

**Location:** `slides/lesson_51_evaluating_llms.md`

# LLM chatbots demo

[![Build and Deploy Documentation](https://github.com/gperdrizet/llms-demo/actions/workflows/docs.yml/badge.svg)](https://github.com/gperdrizet/llms-demo/actions/workflows/docs.yml)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1.2-1C3C3C?logo=langchain&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FF6B00?logo=huggingface&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-UI-FF7C00?logo=gradio&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-black?logo=ollama&logoColor=white)
![llama.cpp](https://img.shields.io/badge/llama.cpp-GGUF-green)
![PostgreSQL](https://img.shields.io/badge/pgvector-PostgreSQL-4169E1?logo=postgresql&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vector%20store-E8572A)

## Introduction

This repository provides hands-on examples and learning resources for working with large language models (LLMs) in local development environments.

### Topics covered

- Local inference with Ollama and llama.cpp
- Direct model loading with HuggingFace Transformers
- LangChain: prompt templates, output parsers, chains, and agents
- RAG (Retrieval-Augmented Generation) with pgvector
- Gradio web interfaces
- Prompting techniques: zero-shot, few-shot, chain-of-thought, ReAct

### Resources included

- **9 demos**: chatbots, LangChain patterns, agents, RAG knowledge systems, fine-tuning & evaluation
- **8 slide decks**: covering deployment, prompting, LangChain, fine-tuning, and evaluation
- **7 activities**: hands-on exercises building on each demo

---

## Documentation

**Complete documentation:** [https://gperdrizet.github.io/llms-demo](https://gperdrizet.github.io/llms-demo/)

The documentation covers:
- Setup and installation
- Demo usage and concepts
- Inference server configuration
- Library reference with code examples
- Model specifications and serving commands
- Systemd deployment for production use
- Slide and activity guides

---

## Quickstart

### 1. Fork and clone

1. Click **Fork** in the top-right corner of this repo on GitHub to create your own copy.
2. Clone your fork:

   ```bash
   git clone https://github.com/<your-username>/llms-demo.git
   ```

### 2. Open in a dev container

1. Open the cloned folder in VS Code.
2. When prompted **"Reopen in Container"**, click it - or run the command **Dev Containers: Reopen in Container** from the Command Palette (`Ctrl+Shift+P`).
3. VS Code will build and start the container. This takes a few minutes the first time.

### 3. What happens during container startup

The dev container is based on the `gperdrizet/llms-gpu` image (NVIDIA GPU-enabled). On first creation, the `postCreateCommand` runs automatically and does the following:

| Step | What it does |
|------|-------------|
| `mkdir -p models/hugging_face && mkdir -p models/ollama` | Creates local directories for model storage |
| `pip install -r requirements.txt` | Installs Python dependencies: **bert-score**, **evaluate**, **gradio**, **huggingface-hub**, **langchain-ollama**, **openai**, **peft**, **python-dotenv**, **trl**, **torch**, **transformers** |
| `bash .devcontainer/install_ollama.sh` | Downloads and installs the Ollama CLI |

The container also pre-configures the following:

| Setting | Detail |
|---------|--------|
| **GPU access** | All host GPUs are passed through (`--gpus all`) |
| **Python interpreter** | `/usr/bin/python` is set as the default |
| **`HF_HOME`** | Points to `models/hugging_face` so Hugging Face downloads stay in the repo |
| **`OLLAMA_MODELS`** | Points to `models/ollama` so Ollama downloads stay in the repo |
| **Port 7860** | Forwarded automatically for Gradio web UIs |
| **VS Code extensions** | Python, Jupyter, Code Spell Checker, and Marp (slide viewer) are installed |

Once the container is ready you can start running the demos - no extra setup needed.

---

## Running the demos

See the [Demos documentation](https://gperdrizet.github.io/llms-demo/demos.html) for detailed instructions on running each chatbot, including:
- Concepts covered in each demo
- Tools and libraries used
- Step-by-step setup and execution

**Quick example** - Ollama chatbot:
```bash
# 1. Start the Ollama server
ollama serve

# 2. Pull a model (in another terminal)
ollama pull qwen2.5:3b

# 3. Run the chatbot
python demos/chatbots/ollama_chatbot.py
```

For complete instructions on all four demos, visit the [documentation](https://gperdrizet.github.io/llms-demo/).



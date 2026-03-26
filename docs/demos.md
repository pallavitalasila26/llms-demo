# Demos

This repository includes six chatbot demos that demonstrate different approaches to local LLM inference. Each demo covers specific concepts and tools.

## Demo 1: HuggingFace chatbot

**File:** `demos/chatbots/huggingface_chatbot.py`

**Concepts covered:**
- Direct model loading (no inference server)
- Chat templates and tokenization
- Generation parameters (temperature, max tokens)
- Decoding and response formatting

**Tools used:**
- [HuggingFace Transformers](libraries.md) - Model loading and inference
- PyTorch - Underlying tensor operations

**Running the demo:**

```bash
# 1. Run the chatbot (downloads model on first run)
python demos/chatbots/huggingface_chatbot.py

# Note: This loads the model directly into memory (no inference server needed).
# First run will download approximately 6GB of model files to models/hugging_face/
```

## Demo 2: Ollama chatbot

**File:** `demos/chatbots/ollama_chatbot.py`

**Concepts covered:**
- Using a local inference server (Ollama)
- Structured message types (SystemMessage, HumanMessage, AIMessage)
- Conversation history management
- Terminal-based interaction

**Tools used:**
- [Ollama](inference_servers.md) - Local inference server
- [LangChain](libraries.md) - LLM application framework

**Running the demo:**

```bash
# 1. Start the Ollama server in a terminal
ollama serve

# 2. Pull a model (in another terminal)
ollama pull qwen2.5:3b

# 3. Run the chatbot
python demos/chatbots/ollama_chatbot.py
```

## Demo 3: llama.cpp chatbot

**File:** `demos/chatbots/llamacpp_chatbot.py`

**Concepts covered:**
- Running large MoE models (120B+ parameters) on consumer hardware
- CPU/GPU memory split for expert layers
- OpenAI-compatible API usage
- Remote vs. local inference servers

**Tools used:**
- [llama.cpp](inference_servers.md) - High-performance C++ inference engine
- OpenAI Python client - Standard API interface

**Running the demo:**

You have two choices: use the hosted model at `gpt.perdrizet.org`, or run llama.cpp locally.

**Option 1: Using the remote server (recommended for quick start)**

```bash
# 1. Create a .env file with your API credentials
cp .env.example .env

# 2. Edit .env and set:
#    PERDRIZET_URL=gpt.perdrizet.org
#    PERDRIZET_API_KEY=your-api-key-here

# 3. Run the chatbot
python demos/chatbots/llamacpp_chatbot.py
```

**Option 2: Running llama.cpp locally**

```bash
# 1. Download a GGUF model (e.g., GPT-OSS-20B)
python utils/download_gpt_oss_20b.py

# 2. Build llama.cpp with CUDA support
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)
cd ..

# 3. Start the llama-server (see model-specific commands in the Models section)
llama.cpp/build/bin/llama-server -m <model.gguf> <flags...>

# 4. Run the chatbot (in another terminal)
python demos/chatbots/llamacpp_chatbot.py
```

> **Note**: For localhost, the defaults work automatically (localhost:8502 with "dummy" API key). For remote servers, configure `PERDRIZET_URL` and `PERDRIZET_API_KEY` in your `.env` file.

## Demo 4: Gradio chatbot

**File:** `demos/chatbots/gradio_chatbot.py`

**Concepts covered:**
- Web-based chat interfaces
- Multi-backend architecture (switching between Ollama/llama.cpp)
- System prompt customization
- Error handling and user feedback

**Tools used:**
- [Gradio](libraries.md) - Rapid UI prototyping
- [LangChain](libraries.md) - LLM orchestration
- [Ollama](inference_servers.md) - Default backend

**Running the demo:**

```bash
# 1. Start the Ollama server in a terminal
ollama serve

# 2. Pull a model (in another terminal)
ollama pull qwen2.5:3b

# 3. Run the Gradio chatbot
python demos/chatbots/gradio_chatbot.py

# 4. Open the URL shown in the terminal (usually http://127.0.0.1:7860)
```

## Demo 5: LangChain basics

**File:** `demos/langchain_patterns/langchain_demo.py`

**Concepts covered:**
- Chat models and LLM wrappers
- Prompt templates with variable substitution
- Structured output parsing with Pydantic schemas
- Basic chains and composition with LCEL
- Few-shot learning patterns

**Tools used:**
- [LangChain](libraries.md) - Core framework components
- [Ollama](inference_servers.md) or [llama.cpp](inference_servers.md) - Backend LLM
- [Gradio](libraries.md) - Interactive web interface

**Running the demo:**

```bash
# 1. Start the Ollama server in a terminal
ollama serve

# 2. Pull a model (in another terminal)
ollama pull qwen2.5:3b

# 3. Run the LangChain demo
python demos/langchain_patterns/langchain_demo.py

# 4. Open the URL shown in the terminal (usually http://127.0.0.1:7860)
```

**Four interactive examples:**

1. **Simple chain**: Prompt template → LLM → String output
   - Try: "machine learning", "photosynthesis", "blockchain"

2. **Sentiment analysis**: Structured JSON output with Pydantic schema
   - Try: Product reviews, comments, social media posts
   - See how the parser extracts sentiment, confidence, and key phrases

3. **Entity extraction**: Different schemas for different entity types
   - Person: name, age, occupation, location
   - Recipe: name, cuisine, ingredients, difficulty
   - Switch schemas to see how the same chain extracts different information

4. **Few-shot learning**: Style classification with examples
   - The model learns from 4 in-prompt examples
   - Try: Technical, casual, formal, or creative writing styles

**What to observe:**
- **Reusability**: Same chain works for multiple inputs
- **Type safety**: Pydantic schemas ensure structured outputs
- **Composability**: Chains combine prompt, model, and parser seamlessly
- **Format instructions**: See how Pydantic schemas generate parsing guidance

## Demo 6: ReAct agent chatbot

**Files:** 
- `demos/langchain_patterns/react_agent_chatbot.py` - Uses LangChain's agent framework
- `demos/langchain_patterns/react_agent_chatbot_manual.py` - Manual implementation from scratch

**Concepts covered:**
- ReAct (Reasoning + Acting) agent pattern
- Multi-step reasoning with tool use
- Tool selection and execution
- Agent iteration loops and error handling
- Comparing high-level frameworks vs. manual implementation

**Tools used:**
- [LangChain](libraries.md) - Agent framework and tool integration
- [Ollama](inference_servers.md) or [llama.cpp](inference_servers.md) - Backend LLM
- [Gradio](libraries.md) - Web interface with reasoning visualization

**Two versions available:**

This demo includes both a production-ready implementation and an educational version that reveals the inner workings:

1. **Built-in agent** (`react_agent_chatbot.py`): Uses LangChain's `create_agent()` API for automatic ReAct pattern handling. This is the recommended approach for real applications.

2. **Manual implementation** (`react_agent_chatbot_manual.py`): A hand-coded ReAct loop with regex parsing that shows exactly what LangChain does behind the scenes. This version demonstrates:
   - How to prompt the LLM to follow the ReAct pattern
   - Parsing LLM responses to extract actions and answers
   - Manual tool execution and observation injection
   - The explicit iteration loop that drives the agent
   
   Use this version to understand the mechanics of agent frameworks before relying on them.

**Running the demo:**

**Version 1: Built-in agent (recommended for beginners)**

```bash
# 1. Start the Ollama server in a terminal
ollama serve

# 2. Pull a model (in another terminal)
ollama pull qwen2.5:3b

# 3. Run the ReAct agent chatbot
python demos/langchain_patterns/react_agent_chatbot.py

# 4. Open the URL shown in the terminal (usually http://127.0.0.1:7860)
```

**Version 2: Manual implementation (educational)**

```bash
# Same setup as Version 1, but run:
python demos/langchain_patterns/react_agent_chatbot_manual.py

# This version shows explicit Thought → Action → Observation cycles
```

**Try these example questions:**
- "How many days until Christmas from today?"
- "Calculate 15% tip on a $47.50 bill"
- "I was born on March 15, 1990. How old am I in days?"
- "What's 25% of 360, divided by 3?"
- "How many weeks between today and New Year's Day 2027?"

**What to observe:**
- Watch the **Reasoning Process** panel (right side) to see how the agent thinks
- Notice when it decides to use tools vs. when it can answer directly
- See the Thought → Action → Observation loop in action
- Try asking multi-step questions that require multiple tool calls
- **Compare both versions**: Run the same question through both demos to see how the manual implementation exposes the mechanics that LangChain handles automatically

## Demo 7: RAG system

**File:** `demos/rag_system/rag_demo.py`

**Concepts covered:**
- Retrieval-Augmented Generation (RAG) pipeline
- Document ingestion, chunking, and embedding
- Vector similarity search with pgvector
- Grounded LLM responses with source citations
- Modular ingestor pattern (`BaseIngestor`)

**Tools used:**
- [LangChain](libraries.md) - RAG chain composition and retriever
- [HuggingFace](libraries.md) - Local embedding model (`all-MiniLM-L6-v2`)
- PostgreSQL + pgvector - Vector store
- [Ollama](inference_servers.md) or [llama.cpp](inference_servers.md) - Backend LLM
- [Gradio](libraries.md) - Web interface with Ingest / Query / Settings tabs

**Running the demo:**

```bash
# 1. Ensure PostgreSQL with pgvector is accessible and .env is configured
#    (DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME)

# 2. Start your LLM backend (llama.cpp is the default)
ollama serve   # or start llama-server

# 3. Run the RAG demo
python demos/rag_system/rag_demo.py

# 4. Open the URL shown in the terminal (usually http://127.0.0.1:7860)
```

**Three tabs:**

1. **Ingest**: Choose a source (Wikipedia), enter a topic, and click **Ingest** to embed and store chunks in the knowledge base
2. **Query**: Ask questions - the retriever finds the most relevant chunks and passes them as context to the LLM
3. **Settings**: Switch between Ollama and llama.cpp backends; clear the vector store collection

**What to observe:**
- The **Sources** panel shows which document chunks were retrieved for each answer
- Ingest the same topic twice to see deduplication behaviour
- Ask a question about something *not* ingested - notice how the grounded answer differs from a hallucinated one
- Switch backends (Ollama vs. llama.cpp) to compare answer quality

## Demo 8: Fine-tuning and alignment demo

**File:** `demos/finetuning/finetuning_demo.py`

**Concepts covered:**
- Behavioral difference between a base model and its instruction-tuned counterpart
- How the chat template links fine-tuning format to inference format
- What SFT and DPO training data actually looks like (Alpaca JSON, ChatML, DPO preference pairs)

**Tools used:**
- [HuggingFace Transformers](libraries.md) - Direct model loading for both base and instruct checkpoints
- [PEFT](libraries.md) / [TRL](libraries.md) - Referenced in the companion activity
- [Gradio](libraries.md) - Two-tab interactive interface

**Running the demo:**

```bash
# Models are downloaded from HuggingFace on first run (~500 MB each).
# Set HF_HOME to control the cache directory.

python demos/finetuning/finetuning_demo.py

# Open the URL shown in the terminal (usually http://127.0.0.1:7860)
```

**Two tabs:**

1. **Model comparison**: The same prompt is sent to `Qwen/Qwen2.5-0.5B` (base, raw text completion) and `Qwen/Qwen2.5-0.5B-Instruct` (instruction-tuned, chat template) simultaneously - responses shown side by side
2. **Dataset formatter**: Enter an instruction and ideal output; see it formatted as Alpaca JSON, ChatML, and DPO preference pairs

**What to observe:**
- On the **completion trap** prompt (`Things I need from the grocery store: 1. Milk 2. Eggs 3.`) - the base model continues the list; the instruct model responds to the intent
- The **Sources** column in the model table shows the actual HuggingFace checkpoint IDs - these are genuinely different weight files, not aliases
- The **chat template** in Tab 2 shows exactly the format the instruct model was trained on: `<|im_start|>system ... <|im_end|>` tokens are what distinguishes the two checkpoints at the data level

## Demo 9: LLM evaluation demo

**File:** `demos/evaluation/evaluation_demo.py`

**Concepts covered:**
- Automated text metrics: ROUGE-1/2/L, BLEU, and BERTScore
- Standardised multiple-choice benchmarking (MMLU-style)
- LLM-as-judge rubric scoring with structured JSON output
- Limitations of each approach and when to use each one

**Tools used:**
- [HuggingFace `evaluate`](libraries.md) - Unified metric computation API
- [`bert-score`](libraries.md) - Contextual embedding similarity
- [LangChain](libraries.md) / [Ollama](inference_servers.md) - Local LLM for benchmark and judge tabs
- [Gradio](libraries.md) - Three-tab interactive interface

**Running the demo:**

```bash
# 1. Install evaluation dependencies
pip install evaluate bert-score

# 2. Start the Ollama server in a terminal
ollama serve

# 3. Pull the benchmark/judge model (in another terminal)
ollama pull qwen2.5:3b

# 4. Run the evaluation demo
python demos/evaluation/evaluation_demo.py

# 5. Open the URL shown in the terminal (usually http://127.0.0.1:7860)
```

*Note: BERTScore downloads a ~400 MB BERT model on first use. Subsequent runs are instant.*

**Three tabs:**

1. **Metric calculator**: Enter a reference and candidate text; compute ROUGE-1/2/L, BLEU, and BERTScore F1 in one click. Pre-filled example illustrates the paraphrase problem.
2. **Mini benchmark**: Run `qwen2.5:3b` against 10 MMLU-style questions (Science, History, Math, Coding); filter by category; see per-question pass/fail and a category breakdown.
3. **LLM-as-judge**: Score a candidate answer on a 1-5 rubric (factual accuracy, relevance, completeness); the judge returns structured JSON parsed into a formatted score table.

**What to observe:**
- In Tab 1: compare exact match, paraphrase, and factual error pairs — ROUGE and BERTScore diverge on the paraphrase (same meaning, different words)
- In Tab 2: the model is instructed to reply with a single letter; see how it handles this constraint
- In Tab 3: try the pre-filled "seasons misconception" answer — does the judge detect the factual error?
- In Tab 3: compare the concise and padded candidate answers from the activity — does the judge exhibit verbosity bias?

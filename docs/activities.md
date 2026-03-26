# Activities

This section contains hands-on activities for practicing prompting techniques and LLM application development.

> **Note:** Activity files are available in the [GitHub repository](https://github.com/gperdrizet/llms-demo/tree/main/activities) and include detailed instructions and starter code.

## Activity 1: LLM word problems

Practice basic prompting techniques and chain-of-thought reasoning by solving word problems with an LLM using the Gradio web interface.

**Duration:** 30-45 minutes

**Skills practiced:**
- Basic prompting strategies
- Chain-of-thought reasoning
- System prompt experimentation
- Comparing different prompting approaches

**Prerequisites:**
- Completed [Quickstart](quickstart.md) setup
- Familiarity with Lesson 46 (Prompting fundamentals)

**What you'll use:**
- Gradio chatbot web interface (`demos/chatbots/gradio_chatbot.py`)
- System prompt customization

**Location:** `activities/activity_1_word_problems.md`

## Activity 2: Text summarization

Build a practical text summarization script applying various prompting techniques.

**Duration:** 45-60 minutes

**Skills practiced:**
- Text preprocessing and chunking
- Prompt engineering for summarization
- Handling long documents
- Iterative refinement

**Prerequisites:**
- Completed Activity 1
- Python programming experience
- Understanding of basic file I/O

**Location:** `activities/activity_2_text_summarization.md`

## Activity 3: Building LangChain chains

Build practical LangChain applications using prompt templates, output parsers, and chains.

**Duration:** 60-75 minutes

**Skills practiced:**
- Creating reusable prompt templates with variables
- Structured data extraction with Pydantic schemas
- JSON parsing with JsonOutputParser
- Composing multi-step chains
- Error handling and debugging chains

**Prerequisites:**
- Completed Activities 1 and 2
- Python programming experience
- Understanding of Lesson 48 (LangChain basics)
- Familiarity with Demo 5 (LangChain demo)

**What you'll do:**
- Build a template-based translator chain
- Create a structured book information extractor
- Compose a multi-step review analysis pipeline
- Learn debugging and best practices

**Location:** `activities/activity_3_langchain_chains.md`

## Activity 4: Extending the ReAct agent

Enhance the ReAct agent chatbot by adding custom tools and testing multi-step reasoning.

**Duration:** 45-60 minutes

**Skills practiced:**
- Creating LangChain tools with the `@tool` decorator
- Understanding agent decision-making and tool selection
- Debugging tool execution and agent behavior
- Multi-step problem solving with tool chaining

**Prerequisites:**
- Completed Activities 1-3
- Python programming experience
- Understanding of Lesson 49 (LangChain advanced - agents)
- Familiarity with the ReAct agent demo

**What you'll do:**
- Study existing tool implementations
- Create a new tool (temperature converter, text analyzer, or custom)
- Register your tool with the agent
- Test single-tool and multi-tool reasoning
- Debug and improve your implementation

**Location:** `activities/activity_4_react_agent_tools.md`

## Activity 5: Extending the RAG knowledge system

Add a new document source to the RAG demo by implementing the `BaseIngestor` interface.

**Duration:** 45-60 minutes

**Skills practiced:**
- Implementing a Python interface (`BaseIngestor`)
- Using LangChain document loaders (`WebBaseLoader`)
- Chunking documents with `RecursiveCharacterTextSplitter`
- Registering new components in a running Gradio application

**Prerequisites:**
- Completed Activities 1-4
- Python programming experience
- Understanding of Lesson 49 (LangChain advanced - RAG pipeline)
- Familiarity with the RAG system demo

**What you'll do:**
- Explore the existing `WikipediaIngestor` and `BaseIngestor` interface
- Implement `URLIngestor` using `WebBaseLoader`
- Register it in the demo and verify it appears in the UI

**Location:** `activities/activity_5_rag_sources.md`


## Activity 6: LoRA fine-tuning experiment

Fine-tune a small language model on a custom instruction dataset using LoRA, then compare its behaviour against the original base model.

**Duration:** 60-90 minutes

**Skills practiced:**
- Loading and prompting base models directly with `transformers`
- Building a small SFT dataset formatted as ChatML
- Configuring and attaching a LoRA adapter with `peft`
- Training with `SFTTrainer` from `trl`
- Comparing base vs. fine-tuned outputs and measuring adapter size

**Prerequisites:**
- Completed Activities 1-5
- Python programming experience
- Understanding of Lesson 50 (Fine-tuning, RLHF, and model alignment)
- Familiarity with the fine-tuning demo (`demos/finetuning/finetuning_demo.py`)

**What you'll do:**
- Run test prompts through the base model to record a baseline
- Write 10-20 `(instruction, response)` pairs in a consistent style of your choosing
- Format the dataset as ChatML and load it with HuggingFace `datasets`
- Attach a LoRA adapter (`r=8`, targeting `q_proj`/`v_proj`) and train for ~5 epochs
- Compare fine-tuned responses to the baseline using the same test prompts
- Save the adapter, measure its size, and reload it via `PeftModel.from_pretrained()`

**Extension challenges:** rank sweep (r=8/16/32), overfitting demonstration (20 epochs), expanding target modules, and `merge_and_unload()` to export a standalone checkpoint.

**Location:** `activities/activity_6_finetuning.md`

## Activity 7: Evaluating LLM outputs

Measure LLM output quality using automated text metrics and an LLM-as-judge rubric, then interpret the results critically.

**Duration:** 45-60 minutes

**Skills practiced:**
- Computing ROUGE-1/2/L, BLEU, and BERTScore with HuggingFace `evaluate`
- Interpreting metric scores and understanding their limitations
- Designing a threshold-based quality gate
- Writing a structured LLM-as-judge rubric prompt
- Parsing structured JSON output from an LLM
- Identifying verbosity bias and prompt sensitivity in judge systems

**Prerequisites:**
- Completed Activities 1-6
- Python programming experience
- Understanding of Lesson 51 (Benchmarking and evaluating LLMs)
- Familiarity with the evaluation demo (`demos/evaluation/evaluation_demo.py`)

**What you'll do:**
- Compute metrics on three text pairs (exact match, paraphrase, factual error) and compare results
- Build a quality gate function using ROUGE and BERTScore thresholds
- Implement an LLM-as-judge function that returns rubric scores as parsed JSON
- Evaluate three candidate answers across factual accuracy, relevance, and completeness
- Detect verbosity bias by comparing concise vs. padded answers

**Location:** `activities/activity_7_evaluation.md`


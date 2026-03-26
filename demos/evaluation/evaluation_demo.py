"""LLM evaluation demo

Demonstrates three complementary approaches to evaluating language model outputs:
automated text metrics, standardized benchmarks, and LLM-as-judge scoring.

Tabs:
1. Metric calculator  - compute ROUGE-1/2/L, BLEU, and BERTScore for any text pair
2. Mini benchmark     - run a local model against 10 MMLU-style questions in real time
3. LLM-as-judge       - score a candidate answer against a rubric using a local model

Usage:
    # Install dependencies if not already present
    pip install evaluate bert-score absl-py rouge-score

    # Ensure Ollama is running with the model pulled, e.g.:
    #   ollama serve
    #   ollama pull qwen2.5:3b

    python demos/evaluation/evaluation_demo.py
"""

import functools
import gradio as gr
from openai import OpenAI

from metrics import compute_metrics
from benchmark import run_benchmark, BENCHMARK_QUESTIONS
from judge import judge_answer, JUDGE_SYSTEM_PROMPT


# --- Configuration ---

# Ollama exposes an OpenAI-compatible API at localhost:11434/v1
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "qwen2.5:3b"

# --- Ollama client (via OpenAI-compatible endpoint) ---

client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama",  # Ollama does not check the key; any non-empty string works
    timeout=120.0,
)

# Bind client and model into the functions so Gradio only sees the UI inputs
_run_benchmark = functools.partial(run_benchmark, client, OLLAMA_MODEL)
_judge_answer = functools.partial(judge_answer, client, OLLAMA_MODEL)

BENCH_MODEL = OLLAMA_MODEL
JUDGE_MODEL = OLLAMA_MODEL


# --- Build Gradio UI ---

with gr.Blocks(title="LLM evaluation demo") as demo:

    gr.Markdown("""
    # LLM evaluation demo

    **Lesson 51 - Benchmarking and evaluating LLMs**

    Three approaches to measuring model quality:
    - **Metric calculator** - automated text overlap and semantic similarity metrics
    - **Mini benchmark** - run a local model against structured knowledge questions
    - **LLM-as-judge** - use a model to score another model's output against a rubric
    """)

    with gr.Tabs():

        # -- Tab 1: Metric Calculator ------------------------------------------
        with gr.Tab("1. Metric calculator"):

            gr.Markdown("""
            ### ROUGE, BLEU, and BERTScore

            Enter a reference text (the ideal answer) and a candidate text (the model output).
            The metrics are computed and displayed in a comparison table.

            | Metric | Type | Best for |
            |--------|------|---------|
            | **ROUGE-1/2/L** | N-gram overlap | Summarization; requires reference |
            | **BLEU** | N-gram precision | Translation; requires reference |
            | **BERTScore** | Semantic similarity | Any task; catches paraphrases |

            *BERTScore loads a BERT model on first use (~400 MB). Subsequent runs are instant.*
            """)

            with gr.Row():
                with gr.Column():
                    ref_input = gr.Textbox(
                        label="Reference text (gold standard)",
                        lines=5,
                        value=(
                            "The Eiffel Tower was built between 1887 and 1889 "
                            "as the entrance arch for the 1889 World's Fair in Paris. "
                            "It was designed by engineer Gustave Eiffel."
                        ),
                    )
                    cand_input = gr.Textbox(
                        label="Candidate text (model output to evaluate)",
                        lines=5,
                        value=(
                            "Paris's iconic iron lattice tower was constructed in the late 1880s "
                            "to serve as the gateway for a World's Fair. "
                            "The structure was engineered by Gustave Eiffel."
                        ),
                    )
                    metric_btn = gr.Button("Compute metrics", variant="primary")

                with gr.Column():
                    metric_output = gr.Markdown(label="Results")

            metric_btn.click(
                fn=compute_metrics,
                inputs=[ref_input, cand_input],
                outputs=[metric_output],
            )

            gr.Markdown("""
            ---

            ### Try these example pairs to see where metrics agree and disagree

            **1. Exact match** - paste the reference as the candidate. All scores should be ~1.0.

            **2. Paraphrase** (pre-filled above) - same meaning, different words.
            - ROUGE: lower (different surface form)
            - BERTScore: high (same meaning)

            **3. Factual error with good fluency** - try:
            > Candidate: *"The Eiffel Tower was built in 1901 as part of a World's Fair in London, designed by engineer Pierre Eiffel."*
            - ROUGE: moderate (many shared words)
            - BERTScore: **high (~0.9)** - the sentence is fluent and topically similar; BERTScore does not detect factual swaps like wrong dates, cities, or names

            **4. Completely wrong answer** - try:
            > Candidate: *"The sky is blue and grass is green."*
            - All metrics: near zero

            **Key insight:** no metric detects factual errors reliably.
            ROUGE misses paraphrases. BERTScore misses subtle factual swaps.
            Use both - and add human review for high-stakes outputs.
            """)

        # -- Tab 2: Mini Benchmark --------------------------------------------
        with gr.Tab("2. Mini benchmark"):

            gr.Markdown(f"""
            ### Local model benchmark

            10 multiple-choice questions across 4 categories, run against `{BENCH_MODEL}` in real time.
            This is a minimal version of the MMLU-style evaluation used in open leaderboards.

            Questions cover: Science, History, Math, and Coding.
            Expected runtime: 30-60 seconds depending on hardware.
            """)

            category_filter = gr.Dropdown(
                choices=["All", "Science", "History", "Math", "Coding"],
                value="All",
                label="Filter by category",
            )
            bench_btn = gr.Button("Run benchmark", variant="primary")
            bench_output = gr.Markdown(label="Results")

            bench_btn.click(
                fn=_run_benchmark,
                inputs=[category_filter],
                outputs=[bench_output],
            )

            gr.Markdown("""
            ---

            ### What to observe

            - Does the model perform evenly across categories, or is one category noticeably weaker?
            - Try a category in isolation - is the weakness consistent?
            - The questions are straightforward factual queries (similar to MMLU easy tier).
              Scores close to 100% are expected for a 3B model on this set.
            - **Prompt sensitivity:** the model is instructed to respond with a single letter.
              Some models ignore this and give full explanations - the scorer handles this by
              extracting the first character. Change `letter = raw[0]` in the code to see
              raw model output.
            """)

        # -- Tab 3: LLM-as-Judge ----------------------------------------------
        with gr.Tab("3. LLM-as-judge"):

            gr.Markdown(f"""
            ### Rubric-based scoring with `{JUDGE_MODEL}`

            Enter a question, the reference (correct) answer, and a candidate answer to evaluate.
            The judge model scores on a 1-5 rubric across three dimensions:
            **factual accuracy**, **relevance**, and **completeness**.

            The judge returns structured JSON parsed into a readable score table.
            """)

            with gr.Row():
                with gr.Column():
                    judge_question = gr.Textbox(
                        label="Question",
                        lines=2,
                        value="What causes the seasons on Earth?",
                    )
                    judge_reference = gr.Textbox(
                        label="Reference answer",
                        lines=4,
                        value=(
                            "Earth's seasons are caused by the tilt of its rotational axis "
                            "(about 23.5 degrees) relative to its orbit around the Sun. "
                            "When the Northern Hemisphere is tilted toward the Sun it experiences "
                            "summer; when tilted away it experiences winter."
                        ),
                    )
                    judge_candidate = gr.Textbox(
                        label="Candidate answer (to be scored)",
                        lines=4,
                        value=(
                            "Seasons happen because the Earth gets closer to and farther "
                            "from the Sun during its orbit. When Earth is closer, it is "
                            "summer; when farther, it is winter."
                        ),
                    )
                    judge_btn = gr.Button("Score with judge", variant="primary")

                with gr.Column():
                    judge_output = gr.Markdown(label="Judge scores")

            judge_btn.click(
                fn=_judge_answer,
                inputs=[judge_question, judge_reference, judge_candidate],
                outputs=[judge_output],
            )

            gr.Markdown("""
            ---

            ### Try these candidate answers to explore the rubric

            **Good answer (should score high):**
            > *"Earth's seasons result from the planet's axial tilt of approximately 23.5 degrees.
            > As Earth orbits the Sun, different hemispheres receive more direct sunlight at different
            > times of year, producing summer and winter."*

            **Partially correct (factual error):**
            > *"Seasons are caused by Earth's distance from the Sun. During summer Earth is closer,
            > and in winter it is farther away."* - this is a common misconception. Does the judge catch it?

            **Off-topic:**
            > *"The Moon orbits the Earth approximately every 27 days."*

            ### What to observe

            - Does the judge catch the factual error in the pre-filled candidate?
            - Does the judge give a higher score to a longer answer even if content is the same?
              (verbosity bias)
            - Try running the same candidate twice - do scores vary? (judge instability)
            - The judge prompt includes explicit step-by-step rubric definitions. Try changing
              the prompt in `JUDGE_SYSTEM_PROMPT` and see how scores change. (prompt sensitivity)
            """)

    gr.Markdown("""
    ---

    ## Key takeaways

    1. **ROUGE measures overlap** - fast and reference-based; misses paraphrases
    2. **BERTScore measures meaning** - catches paraphrases; still misses subtle factual errors
    3. **No automated metric detects hallucinations reliably** - they require human review or LLM-as-judge
    4. **Benchmark accuracy measures knowledge breadth** - strong benchmark scores don't guarantee good generation quality
    5. **LLM-as-judge scales human evaluation** but is sensitive to verbosity bias, self-preference, and prompt wording
    """)


if __name__ == "__main__":
    demo.launch()

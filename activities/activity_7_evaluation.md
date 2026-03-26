# Activity 7: Evaluating LLM outputs

**Objective:** Measure LLM output quality using automated text metrics and an LLM-as-judge rubric, then interpret the results critically.

**Duration:** 45-60 minutes

---

## Overview

In this activity you will:

1. Compute ROUGE, BLEU, and BERTScore metrics on reference/candidate text pairs and interpret what the numbers mean
2. Implement a structured LLM-as-judge evaluation function that returns rubric scores as parsed JSON

Automated metrics give you fast, reproducible signals at scale.
LLM-as-judge extends that to dimensions — accuracy, relevance, completeness — that no surface metric can capture.

---

## Setup

### Step 1: Install dependencies

```bash
pip install evaluate bert-score
```

`evaluate` provides a unified API to ROUGE, BLEU, BERTScore, and many other metrics.
`bert-score` is the underlying library used by the `bertscore` evaluate metric.

### Step 2: Create your activity script

Create a file `activities/my_evaluation.py`. All code snippets go into this file.

```python
"""Activity 7: LLM evaluation experiment."""

import json
import evaluate
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
```

### Step 3: Load the metrics once

Loading metrics at module level avoids repeated disk reads:

```python
rouge_metric = evaluate.load("rouge")
bleu_metric  = evaluate.load("bleu")
bert_metric  = evaluate.load("bertscore")
```

BERTScore will download a ~400 MB BERT model on first use and cache it automatically.

---

## Part 1: Automated text metrics

### Step 1: Define a helper function

Write a function that computes all three metric families for a single reference/candidate pair and returns a readable dictionary:

```python
def score_pair(reference: str, candidate: str) -> dict:
    """Return ROUGE-1/2/L, BLEU, and BERTScore F1 for a reference/candidate pair."""

    r = rouge_metric.compute(
        predictions=[candidate],
        references=[reference],
        use_stemmer=True,
    )

    b = bleu_metric.compute(
        predictions=[candidate.split()],
        references=[[reference.split()]],
    )

    bs = bert_metric.compute(
        predictions=[candidate],
        references=[reference],
        lang="en",
        model_type="distilbert-base-uncased",
    )

    return {
        "rouge1":       round(r["rouge1"], 3),
        "rouge2":       round(r["rouge2"], 3),
        "rougeL":       round(r["rougeL"], 3),
        "bleu":         round(b["bleu"], 3),
        "bertscore_f1": round(bs["f1"][0], 3),
    }
```

### Step 2: Build a test set

Create three reference/candidate pairs that each expose a different weakness:

```python
reference = (
    "The Eiffel Tower was built between 1887 and 1889 as the entrance arch "
    "for the 1889 World's Fair in Paris. It was designed by engineer Gustave Eiffel."
)

pairs = {
    "exact_match": (
        reference,
        reference,  # identical copy
    ),
    "paraphrase": (
        reference,
        "Paris's iconic iron lattice tower was constructed in the late 1880s "
        "to serve as a gateway for a World's Fair. "
        "The structure was engineered by Gustave Eiffel.",
    ),
    "factual_error": (
        reference,
        "The Eiffel Tower was completed in 1901 as the centrepiece of a World's "
        "Fair held in London. Its designer was Pierre Eiffel.",
    ),
}
```

### Step 3: Run and print the results

```python
print(f"{'Pair':<15}  {'R-1':>6}  {'R-2':>6}  {'R-L':>6}  {'BLEU':>6}  {'BERT':>6}")
print("-" * 55)

for name, (ref, cand) in pairs.items():
    scores = score_pair(ref, cand)
    print(
        f"{name:<15}  "
        f"{scores['rouge1']:>6.3f}  "
        f"{scores['rouge2']:>6.3f}  "
        f"{scores['rougeL']:>6.3f}  "
        f"{scores['bleu']:>6.3f}  "
        f"{scores['bertscore_f1']:>6.3f}"
    )
```

Expected output (approximate):

```
Pair             R-1     R-2     R-L    BLEU    BERT
-------------------------------------------------------
exact_match     1.000   1.000   1.000  1.000   1.000
paraphrase      0.500   0.240   0.440  0.070   0.910
factual_error   0.610   0.370   0.430  0.140   0.820
```

**Questions to consider:**

- Which metric best separates the paraphrase from the exact match?
- Why does BERTScore give a high score to the factual error pair? What does this tell you about what BERTScore actually measures?
- Add a fourth pair: a completely unrelated sentence as the candidate (e.g. `"The sky is blue."`). What does each metric return?
- What task would BLEU be best suited to? What does the low score on the paraphrase tell you?

### Step 4: Threshold exercise

Suppose you are building a summarization pipeline and want to flag low-quality outputs for human review. Write a function that takes a `scores` dict and returns `"PASS"` or `"FLAG"` based on thresholds:

```python
def quality_gate(scores: dict, thresholds: dict) -> str:
    """Return PASS or FLAG depending on whether all thresholds are met.

    Args:
        scores:     Output of score_pair().
        thresholds: Dict of metric name → minimum acceptable value.

    Returns:
        'PASS' if all thresholds are met, 'FLAG' otherwise.
    """
    for metric, minimum in thresholds.items():
        if scores[metric] < minimum:
            return "FLAG"
    return "PASS"

thresholds = {"rouge1": 0.4, "bertscore_f1": 0.85}

for name, (ref, cand) in pairs.items():
    scores = score_pair(ref, cand)
    result = quality_gate(scores, thresholds)
    print(f"{name:<15}  {result}")
```

Try adjusting the thresholds until the `factual_error` pair is flagged but the `paraphrase` passes.

---

## Part 2: LLM-as-judge

### Step 1: Write the judge system prompt

A well-structured rubric prompt is the most important ingredient in a reliable LLM-as-judge evaluation:

```python
JUDGE_SYSTEM_PROMPT = """\
You are an objective evaluator of language model answers.
You will be given a question, a reference answer, and a candidate answer.
Score the candidate on the rubric below and return your response as JSON.

Rubric:
- factual_accuracy (1-5): Does the answer contain only correct information?
  5=completely accurate, 4=one minor error, 3=partially correct,
  2=mostly wrong, 1=completely wrong
- relevance (1-5): Does the answer address what was asked?
  5=directly on-topic, 3=partially relevant, 1=off-topic
- completeness (1-5): Are all key points covered without padding?
  5=comprehensive, 3=missing some key points, 1=barely covers the topic

Return ONLY valid JSON in this exact format:
{
  "factual_accuracy": <score>,
  "relevance": <score>,
  "completeness": <score>,
  "overall": <average of the three scores rounded to 1 decimal>,
  "reasoning": "<one sentence explaining your scores>"
}"""
```

### Step 2: Implement the judge function

```python
client = ChatOllama(
    model="qwen2.5:3b",
    base_url="http://localhost:11434",
    temperature=0.0,
)

def judge(question: str, reference: str, candidate: str) -> dict:
    """Score a candidate answer using the LLM judge.

    Args:
        question:  The question the candidate is answering.
        reference: The correct reference answer.
        candidate: The answer to be scored.

    Returns:
        Parsed rubric scores as a dictionary.
    """
    user_message = (
        f"Question: {question}\n\n"
        f"Reference answer: {reference}\n\n"
        f"Candidate answer: {candidate}"
    )

    response = client.invoke([
        SystemMessage(content=JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ])

    raw = response.content.strip()

    # Strip markdown fences if the model adds them
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw)
```

### Step 3: Evaluate a set of candidate answers

Use this question and reference to test three different candidate answers:

```python
question = "What causes the seasons on Earth?"

reference = (
    "Earth's seasons are caused by the tilt of its rotational axis "
    "(approximately 23.5 degrees) relative to its orbit around the Sun. "
    "When the Northern Hemisphere is tilted toward the Sun it experiences "
    "summer; when tilted away it experiences winter."
)

candidates = {
    "correct": (
        "Seasons result from Earth's axial tilt of about 23.5 degrees. "
        "As the planet orbits the Sun, different hemispheres receive more "
        "direct sunlight at different times of year."
    ),
    "misconception": (
        "Seasons happen because the Earth moves closer to and farther from "
        "the Sun. In summer Earth is closer to the Sun; in winter it is "
        "farther away."
    ),
    "off_topic": (
        "The Moon completes one orbit around the Earth roughly every 27 days."
    ),
}

for name, candidate in candidates.items():
    scores = judge(question, reference, candidate)
    print(f"\n{name}:")
    print(f"  Factual accuracy:  {scores['factual_accuracy']}")
    print(f"  Relevance:         {scores['relevance']}")
    print(f"  Completeness:      {scores['completeness']}")
    print(f"  Overall:           {scores['overall']}")
    print(f"  Reasoning:         {scores['reasoning']}")
```

**Questions to consider:**

- Does the judge assign a lower factual accuracy score to the `misconception` candidate? Earth's distance from the Sun is a common misconception — how reliably does the judge detect it?
- Does the judge score the `off_topic` candidate correctly across all three dimensions?
- Run the same candidate twice. Do the scores differ? What does this tell you about judge reliability?
- Rewrite the `JUDGE_SYSTEM_PROMPT` without the explicit 1-5 explanations. Do the scores change? What does this tell you about prompt sensitivity?

### Step 4: Detect verbosity bias

Generate one concise and one padded version of the same answer and compare scores:

```python
concise = "Seasons are caused by Earth's axial tilt of 23.5 degrees."

padded = (
    "That is a wonderful and deeply interesting question. "
    "Seasons, as we know them, are a fascinating phenomenon on our planet. "
    "Broadly speaking, they are caused by the fact that Earth's axis is tilted "
    "at an angle of approximately 23.5 degrees relative to its orbital plane. "
    "This means that as our planet makes its way around the Sun each year, "
    "the amount of sunlight reaching each hemisphere varies considerably. "
    "In summary, it is the axial tilt that is the root cause of the seasons."
)

for name, candidate in [("concise", concise), ("padded", padded)]:
    scores = judge(question, reference, candidate)
    print(f"{name}: overall={scores['overall']}, reasoning={scores['reasoning']}")
```

Does the judge give the padded answer a higher completeness score even though it adds no new information? This is verbosity bias — a well-known failure mode in LLM-as-judge systems.

---

## Summary

| Skill | Covered in |
|-------|------------|
| Computing ROUGE-1/2/L | Part 1, Step 1-2 |
| Computing BLEU | Part 1, Step 1-2 |
| Computing BERTScore | Part 1, Step 1-2 |
| Interpreting metric scores | Part 1, Step 3 |
| Designing a quality gate | Part 1, Step 4 |
| Writing a rubric judge prompt | Part 2, Step 1 |
| Parsing structured LLM output | Part 2, Step 2 |
| Evaluating multiple candidates | Part 2, Step 3 |
| Identifying verbosity bias | Part 2, Step 4 |

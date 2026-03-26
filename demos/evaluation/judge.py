"""LLM-as-judge: rubric-based answer scoring using a local model."""

import json
from openai import OpenAI

JUDGE_SYSTEM_PROMPT = """\
You are an objective evaluator of language model answers.
You will be given a question, a reference answer, and a candidate answer to evaluate.
Score the candidate answer on the following rubric and return your response as JSON.

Rubric:
- factual_accuracy (1-5): Does the answer contain only correct, verified information?
  5=completely accurate, 4=one minor error, 3=partially correct, 2=mostly wrong, 1=completely wrong
- relevance (1-5): Does the answer address what was asked?
  5=directly answers the question, 3=partially relevant, 1=off-topic
- completeness (1-5): Is the answer complete without being unnecessarily padded?
  5=covers all key points, 3=missing some important points, 1=barely covers the topic

Return ONLY valid JSON in this exact format:
{
  "factual_accuracy": <score>,
  "relevance": <score>,
  "completeness": <score>,
  "overall": <average of the three scores rounded to 1 decimal>,
  "reasoning": "<one or two sentences explaining your scores>"
}"""


def judge_answer(client: OpenAI, model: str, question: str, reference: str, candidate: str) -> str:
    """Score a candidate answer using an LLM judge with a structured rubric.

    Args:
        client: OpenAI-compatible client pointed at a llama.cpp server.
        model: Model name to pass in the API request.
        question: The original question posed to the model.
        reference: The correct reference answer.
        candidate: The candidate answer to evaluate.

    Returns:
        Formatted markdown showing the rubric scores and reasoning.
    """
    if not question.strip() or not reference.strip() or not candidate.strip():
        return "Please fill in all three fields."

    user_message = f"""\
Question: {question}

Reference answer: {reference}

Candidate answer to evaluate: {candidate}"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if the model wraps the JSON
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        scores = json.loads(raw)

        lines = [
            f"### Overall score: **{scores['overall']} / 5**\n",
            "| Dimension | Score |",
            "|-----------|-------|",
            f"| Factual accuracy | {scores['factual_accuracy']} / 5 |",
            f"| Relevance | {scores['relevance']} / 5 |",
            f"| Completeness | {scores['completeness']} / 5 |",
            f"\n**Judge's reasoning:** {scores['reasoning']}",
        ]
        return "\n".join(lines)

    except json.JSONDecodeError:
        return f"Judge returned non-JSON output:\n\n```\n{response.choices[0].message.content}\n```"

    except Exception as e:
        return f"Error calling judge model: {e}"

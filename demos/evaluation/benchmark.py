"""Mini benchmark: MMLU-style multiple-choice questions and runner."""

from openai import OpenAI

BENCHMARK_QUESTIONS = [
    {
        "category": "Science",
        "question": "Which gas makes up the majority of Earth's atmosphere?",
        "choices": ["A. Oxygen", "B. Carbon dioxide", "C. Nitrogen", "D. Argon"],
        "answer": "C",
    },
    {
        "category": "Science",
        "question": "What is the chemical symbol for gold?",
        "choices": ["A. Go", "B. Gd", "C. Ag", "D. Au"],
        "answer": "D",
    },
    {
        "category": "Science",
        "question": "Which planet has the most known moons?",
        "choices": ["A. Jupiter", "B. Saturn", "C. Neptune", "D. Uranus"],
        "answer": "B",
    },
    {
        "category": "History",
        "question": "In which year did World War II end?",
        "choices": ["A. 1943", "B. 1944", "C. 1945", "D. 1946"],
        "answer": "C",
    },
    {
        "category": "History",
        "question": "Who was the first person to walk on the moon?",
        "choices": ["A. Buzz Aldrin", "B. Yuri Gagarin", "C. Neil Armstrong", "D. John Glenn"],
        "answer": "C",
    },
    {
        "category": "Math",
        "question": "What is the square root of 144?",
        "choices": ["A. 10", "B. 11", "C. 12", "D. 14"],
        "answer": "C",
    },
    {
        "category": "Math",
        "question": "If a rectangle has length 8 cm and width 5 cm, what is its area?",
        "choices": ["A. 13 cm²", "B. 26 cm²", "C. 40 cm²", "D. 80 cm²"],
        "answer": "C",
    },
    {
        "category": "Math",
        "question": "What is 15% of 200?",
        "choices": ["A. 20", "B. 25", "C. 30", "D. 35"],
        "answer": "C",
    },
    {
        "category": "Coding",
        "question": "In Python, which method removes and returns the last element of a list?",
        "choices": ["A. remove()", "B. delete()", "C. pop()", "D. discard()"],
        "answer": "C",
    },
    {
        "category": "Coding",
        "question": "What does the 'len()' function return when called on a string?",
        "choices": [
            "A. The number of words",
            "B. The number of characters",
            "C. The number of bytes",
            "D. The number of lines",
        ],
        "answer": "B",
    },
]

_PROMPT_TEMPLATE = """\
Answer the following multiple-choice question. Reply with ONLY the letter of the correct answer (A, B, C, or D) and nothing else.

Question: {question}
{choices}

Answer:"""


def run_benchmark(client: OpenAI, model: str, category_filter: str) -> str:
    """Run the selected benchmark questions through the model.

    Args:
        client: OpenAI-compatible client pointed at a llama.cpp server.
        model: Model name to pass in the API request.
        category_filter: Category to filter on, or 'All'.

    Returns:
        Markdown results table with per-question pass/fail and a summary.
    """
    questions = BENCHMARK_QUESTIONS
    if category_filter != "All":
        questions = [q for q in questions if q["category"] == category_filter]

    if not questions:
        return "No questions found for that category."

    results = []
    correct = 0

    for i, q in enumerate(questions):
        choices_text = "\n".join(q["choices"])
        prompt = _PROMPT_TEMPLATE.format(
            question=q["question"],
            choices=choices_text,
        )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=4,
            )
            raw = response.choices[0].message.content.strip().upper()
            letter = raw[0] if raw else "?"
            passed = letter == q["answer"]
        except Exception:
            letter = "ERR"
            passed = False

        icon = "PASS" if passed else "FAIL"
        results.append((i + 1, q["category"], q["question"][:60] + "...", q["answer"], letter, icon))
        if passed:
            correct += 1

    lines = [
        f"### Results: {correct}/{len(questions)} correct ({100 * correct // len(questions)}%)\n",
        "| # | Category | Question | Expected | Got | Result |",
        "|---|----------|----------|----------|-----|--------|",
    ]

    category_scores: dict[str, list] = {}
    for num, cat, question, expected, got, icon in results:
        lines.append(f"| {num} | {cat} | {question} | {expected} | {got} | **{icon}** |")
        category_scores.setdefault(cat, []).append(icon == "PASS")

    lines.append("\n**Per-category breakdown:**\n")
    for cat, scores in sorted(category_scores.items()):
        lines.append(f"- **{cat}**: {sum(scores)}/{len(scores)}")

    return "\n".join(lines)

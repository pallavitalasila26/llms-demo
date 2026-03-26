"""Automated text metrics: ROUGE, BLEU, and BERTScore.

Metrics are loaded lazily on first use and cached for subsequent calls.
"""

import threading

BERTSCORE_MODEL_TYPE = "distilbert-base-uncased"

_metrics: dict = {}
_metric_lock = threading.Lock()


def _load_metric(name: str):
    """Load a HuggingFace evaluate metric, caching it after the first load."""
    if name in _metrics:
        return _metrics[name]
    with _metric_lock:
        if name in _metrics:
            return _metrics[name]
        import evaluate
        print(f"Loading metric: {name} ...")
        _metrics[name] = evaluate.load(name)
        print(f"  {name} ready.")
    return _metrics[name]


def compute_metrics(reference: str, candidate: str) -> str:
    """Compute ROUGE-1/2/L, BLEU, and BERTScore for a reference/candidate pair.

    Args:
        reference: The gold-standard reference text.
        candidate: The generated text to evaluate.

    Returns:
        A markdown table of metric scores.
    """
    if not reference.strip() or not candidate.strip():
        return "Please enter both a reference and a candidate text."

    rows = []

    try:
        rouge = _load_metric("rouge")
        r = rouge.compute(
            predictions=[candidate],
            references=[reference],
            use_stemmer=True,
        )
        rows.append(("ROUGE-1", f"{r['rouge1']:.3f}", "Unigram overlap F1"))
        rows.append(("ROUGE-2", f"{r['rouge2']:.3f}", "Bigram overlap F1"))
        rows.append(("ROUGE-L", f"{r['rougeL']:.3f}", "Longest common subsequence F1"))
    except Exception as e:
        rows.append(("ROUGE", "error", str(e)))

    try:
        bleu = _load_metric("bleu")
        b = bleu.compute(
            predictions=[candidate],
            references=[reference],
        )
        rows.append(("BLEU", f"{b['bleu']:.3f}", "N-gram precision (1-4)"))
    except Exception as e:
        rows.append(("BLEU", "error", str(e)))

    try:
        bertscore = _load_metric("bertscore")
        bs = bertscore.compute(
            predictions=[candidate],
            references=[reference],
            lang="en",
            model_type=BERTSCORE_MODEL_TYPE,
        )
        rows.append((
            "BERTScore F1",
            f"{bs['f1'][0]:.3f}",
            "Semantic similarity (contextual embeddings)",
        ))
    except Exception as e:
        rows.append(("BERTScore", "error", str(e)))

    lines = [
        "| Metric | Score | What it measures |",
        "|--------|-------|-----------------|",
    ]
    for metric, score, description in rows:
        lines.append(f"| **{metric}** | `{score}` | {description} |")

    return "\n".join(lines)

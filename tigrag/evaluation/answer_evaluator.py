# -*- coding: utf-8 -*-
"""
answer_judge_parallel.py

run with python3 -m tigrag.evaluation.answer_evaluator

- Align queries between two methods (e.g., lightrag vs tig) per dataset
- For each query, perform one LLM request to evaluate Answer 1 vs Answer 2
- Run requests in parallel using a thread pool (configurable workers)
- Save a single JSON per (dataset, method pair) at a specified directory with:
    {
      "dataset": "...",
      "method_a": "...",
      "method_b": "...",
      "items": [
        {"query": "...", "evaluation": {...}},
        ...
      ],
      "final_eval": {
        "Comprehensiveness": {"method_a": {"count": X, "percent": P}, "method_b": {...}, "total": N},
        "Diversity": {...},
        "Empowerment": {...},
        "Overall": {...}
      }
    }

Input files (must exist):
  <input_dir>/<method>_<dataset>_answers.json
    -> JSON array of {"query": "...", "answer": "..."}

CLI examples:
  python answer_judge_parallel.py --input-dir tigrag/evaluation/answers --output-dir tigrag/evaluation/judgments
  python answer_judge_parallel.py --datasets cooking history --methods lightrag tig --workers 8
"""

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any
import threading
from tigrag.utils.llm_invoker import LLMInvoker  # noqa: E402

# -------------------------------
# Defaults / config
# -------------------------------

DATASET_NAMES = ["cooking", "agriculture", "history"]
DEFAULT_METHODS = ["lightrag", "tig"]

DEFAULT_WORKERS = 8  # number of concurrent requests

LLM_PARAMS = {
    "max_new_tokens": 800,
    "temperature": 0.2,
    "top_p": 0.95,
}

SYSTEM_PROMPT = """\
---
Role
---
You are an expert tasked with evaluating two answers to the same question based on three criteria: Comprehensiveness, Diversity, and Empowerment.
"""

USER_PROMPT_TEMPLATE = """\
You will evaluate two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

- **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
- **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
- **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?

For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these three categories.

Here is the question:
{query}

Here are the two answers:

**Answer 1:**
{answer1}

**Answer 2:**
{answer2}

Evaluate both answers using the three criteria listed above and provide detailed explanations for each criterion.

Output your evaluation in the following JSON format:

{{
    "Comprehensiveness": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Provide explanation here]"
    }},
    "Diversity": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Provide explanation here]"
    }},
    "Empowerment": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Provide explanation here]"
    }},
    "Overall Winner": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Summarize why this answer is the overall winner based on the three criteria]"
    }}
}}
"""

# Thread-local LLM instances (safer if the invoker is not thread-safe)
_thread_local = threading.local()


# -------------------------------
# Basic helpers
# -------------------------------

def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def load_answers(path: str) -> List[dict]:
    """Load JSON array of {'query': str, 'answer': str}."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing answers file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Answers file must be a JSON array: {path}")
    for i, item in enumerate(data):
        if not isinstance(item, dict) or "query" not in item or "answer" not in item:
            raise ValueError(f"Invalid item at index {i} in {path}; expected dict with 'query' and 'answer'.")
    return data


def index_by_query(rows: List[dict]) -> Dict[str, str]:
    """Map query -> answer; enforce a single answer per query."""
    out: Dict[str, str] = {}
    for r in rows:
        q = str(r["query"]).strip()
        if q in out:
            raise ValueError(f"Duplicate query: {q}")
        out[q] = str(r["answer"])
    return out


def align_queries(input_dir: str, dataset: str, method_a: str, method_b: str) -> List[Tuple[str, str, str]]:
    """Return list of (query, answer_a, answer_b) for queries common to both methods."""
    a_path = os.path.join(input_dir, f"{method_a}_{dataset}_answers.json")
    b_path = os.path.join(input_dir, f"{method_b}_{dataset}_answers.json")
    a_rows = load_answers(a_path)
    b_rows = load_answers(b_path)
    a_idx = index_by_query(a_rows)
    b_idx = index_by_query(b_rows)
    common = sorted(set(a_idx.keys()) & set(b_idx.keys()))
    return [(q, a_idx[q], b_idx[q]) for q in common]


def build_user_prompt(query: str, a1: str, a2: str) -> str:
    """Fill prompt template."""
    return USER_PROMPT_TEMPLATE.format(query=query, answer1=a1, answer2=a2)


def _get_thread_llm() -> LLMInvoker:
    """Get/create a thread-local LLMInvoker instance."""
    if getattr(_thread_local, "llm", None) is None:
        _thread_local.llm = LLMInvoker()
    return _thread_local.llm


def _parse_llm_response(resp: Any) -> Any:
    """Robustly parse model output into JSON."""
    # Possible shapes: string JSON; dict with 'content'; OpenAI-like choices; etc.
    text = None
    if isinstance(resp, str):
        text = resp
    elif isinstance(resp, dict):
        if "content" in resp and isinstance(resp["content"], str):
            text = resp["content"]
        elif "choices" in resp and isinstance(resp["choices"], list) and resp["choices"]:
            first = resp["choices"][0]
            if isinstance(first, dict):
                msg = first.get("message")
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    text = msg["content"]
    elif isinstance(resp, list) and resp and isinstance(resp[0], dict) and isinstance(resp[0].get("content"), str):
        text = resp[0]["content"]

    if not isinstance(text, str):
        text = str(resp)

    # Try direct JSON
    try:
        return json.loads(text.strip())
    except Exception:
        # Heuristic extraction of first JSON object
        s = text.strip()
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = s[start:end + 1]
            try:
                return json.loads(candidate)
            except Exception:
                return {"_raw_text": s, "_parse_error": True}
        return {"_raw_text": s, "_parse_error": True}


def judge_once(query: str, ans1: str, ans2: str) -> Dict[str, Any]:
    """Perform a single LLM call to evaluate Answer 1 vs Answer 2 for the query."""
    llm = _get_thread_llm()
    prompt = build_user_prompt(query, ans1, ans2)
    resp = llm(parameters=LLM_PARAMS, message=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ])
    parsed = _parse_llm_response(resp)
    return {"query": query, "evaluation": parsed}


# -------------------------------
# Aggregation of winners
# -------------------------------

def _winner_from_block(block: Any) -> str:
    """Extract 'Winner' field from a criterion block if present."""
    if isinstance(block, dict):
        w = block.get("Winner")
        if isinstance(w, str):
            return w.strip()
    return ""


def _tally_category(items: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    """
    Compute counts and percentages for 'Answer 1' vs 'Answer 2' in a given category key.
    Returns a dict with counts and percents.
    """
    a1 = a2 = 0
    total_valid = 0
    for it in items:
        ev = it.get("evaluation", {})
        if not isinstance(ev, dict):
            continue
        block = ev.get(key)
        winner = _winner_from_block(block)
        if winner == "Answer 1":
            a1 += 1
            total_valid += 1
        elif winner == "Answer 2":
            a2 += 1
            total_valid += 1
        else:
            # ignore unparsable or missing winners for this key
            pass
    pct_a1 = (a1 / total_valid * 100.0) if total_valid > 0 else 0.0
    pct_a2 = (a2 / total_valid * 100.0) if total_valid > 0 else 0.0
    return {
        "method_a": {"count": a1, "percent": round(pct_a1, 2)},
        "method_b": {"count": a2, "percent": round(pct_a2, 2)},
        "total": total_valid,
    }


def build_final_eval(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build final evaluation summary across categories."""
    return {
        "Comprehensiveness": _tally_category(items, "Comprehensiveness"),
        "Diversity": _tally_category(items, "Diversity"),
        "Empowerment": _tally_category(items, "Empowerment"),
        # For overall, models used "Overall Winner" block name in the prompt
        "Overall": _tally_category(items, "Overall Winner"),
    }


def log_final_eval(dataset: str, method_a: str, method_b: str, final_eval: Dict[str, Any]) -> None:
    """Log human-readable summary per category using logging.info."""
    logging.info("=== Final evaluation for dataset '%s' (%s vs %s) ===", dataset, method_a, method_b)
    for cat in ["Comprehensiveness", "Diversity", "Empowerment", "Overall"]:
        stats = final_eval.get(cat, {})
        ma = stats.get("method_a", {})
        mb = stats.get("method_b", {})
        total = stats.get("total", 0)
        logging.info(
            "%s -> %s: %s%% (%s) | %s: %s%% (%s) | total: %s",
            cat,
            method_a, ma.get("percent", 0.0), ma.get("count", 0),
            method_b, mb.get("percent", 0.0), mb.get("count", 0),
            total,
        )


# -------------------------------
# Orchestration
# -------------------------------

def evaluate_dataset_parallel(
    dataset: str,
    method_a: str,
    method_b: str,
    input_dir: str,
    workers: int,
) -> List[Dict[str, Any]]:
    """
    Align queries and evaluate them in parallel.
    Returns a list of {"query": str, "evaluation": dict} in the original sorted order.
    """
    aligned = align_queries(input_dir, dataset, method_a, method_b)
    if not aligned:
        return []

    # Keep original order; collect futures with their index
    items: List[Dict[str, Any]] = [None] * len(aligned)  # type: ignore
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        future_to_idx = {}
        for idx, (q, a1, a2) in enumerate(aligned):
            fut = ex.submit(judge_once, q, a1, a2)
            future_to_idx[fut] = idx

        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                items[idx] = fut.result()
            except Exception as e:
                # Store a parse/error placeholder instead of failing the whole run
                q = aligned[idx][0]
                items[idx] = {"query": q, "evaluation": {"_error": str(e)}}

    # Filter out any gaps (shouldn't happen, but stay safe)
    return [it for it in items if isinstance(it, dict)]


def write_json_output(
    output_dir: str,
    dataset: str,
    method_a: str,
    method_b: str,
    items: List[Dict[str, Any]],
) -> str:
    """Write the final JSON (with per-query evaluations and final_eval) and return the path."""
    ensure_dir(output_dir)
    out_path = os.path.join(output_dir, f"{dataset}__{method_a}_vs_{method_b}.json")
    final_eval = build_final_eval(items)
    payload = {
        "dataset": dataset,
        "method_a": method_a,
        "method_b": method_b,
        "items": items,          # each item: {"query": "...", "evaluation": {...}}
        "final_eval": final_eval # category summary appended as requested
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    # Log overview
    log_final_eval(dataset, method_a, method_b, final_eval)
    return out_path


# -------------------------------
# CLI
# -------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Parallel LLM judging per query with per-category summary.")
    ap.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing <method>_<dataset>_answers.json (e.g., tigrag/evaluation/answers)."
    )
    ap.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write final JSON files (e.g., tigrag/evaluation/judgments)."
    )
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=DATASET_NAMES,
        help=f"Datasets to process (default: {DATASET_NAMES})."
    )
    ap.add_argument(
        "--methods",
        nargs=2,
        default=DEFAULT_METHODS,
        help=f"Exactly two methods to compare (default: {DEFAULT_METHODS})."
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of parallel worker threads (default: {DEFAULT_WORKERS})."
    )
    return ap.parse_args()


def main():
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    # ---- Direct configuration (edit as needed) ----
    input_dir: str = "tigrag/evaluation/answers"       # where <method>_<dataset>_answers.json live
    output_dir: str = "tigrag/evaluation/judgments"     # where results JSON will be written
    datasets: List[str] = ["cooking"] # ["cooking", "agriculture", "history"]
    method_a: str = "lightrag"
    method_b: str = "tigrag"
    workers: int = 1                                     # number of parallel threads
    # ------------------------------------------------

    for ds in datasets:
        logging.info("Starting dataset '%s' (%s vs %s) with %d workers", ds, method_a, method_b, workers)
        try:
            items = evaluate_dataset_parallel(ds, method_a, method_b, input_dir, workers)
            if not items:
                logging.info("No aligned queries for dataset '%s' — skipping.", ds)
                continue
            out_path = write_json_output(output_dir, ds, method_a, method_b, items)
            logging.info("Wrote JSON: %s", out_path)
        except Exception as e:
            logging.exception("Failed dataset '%s': %s", ds, e)


if __name__ == "__main__":
    main()
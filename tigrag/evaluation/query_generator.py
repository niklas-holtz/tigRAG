"""
Generate high-level questions per dataset using an LLM, based on token-based summaries
from each dataset's contexts. Results are saved to: <output_dir>/<dataset>_queries.txt
"""

import os
import re
import sys
import logging
import argparse
from typing import List, Optional
import torch
from transformers import AutoTokenizer
from tigrag.utils.llm_invoker import LLMInvoker
from tigrag.dataset_provider.ultradomain_dataset_provider import UltraDomainDatasetProvider


# -------------------------------
# Logging
# -------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# -------------------------------
# Model / LLM params
# -------------------------------

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

LLM_PARAMS = {
    "max_new_tokens": 4056,
    "top_p": 0.95,
    "temperature": 0.1,
}


# -------------------------------
# Utilities
# -------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    logger.info(f"Ensured output directory exists: {path}")


def get_summary(context: str, tokenizer: AutoTokenizer, total_tokens: int = 200) -> str:
    if not context:
        return ""

    tokens = tokenizer.tokenize(context)
    if len(tokens) <= total_tokens:
        return context.strip()

    half = max(1, total_tokens // 2)

    start_start = min(1000, max(0, len(tokens) - half))
    start_end = min(len(tokens), start_start + half)
    start_tokens = tokens[start_start:start_end]

    if len(tokens) > (half + 1000):
        end_tokens = tokens[-(half + 1000): -1000]
    else:
        end_tokens = tokens[-half:]

    summary_tokens = start_tokens + end_tokens
    summary = tokenizer.convert_tokens_to_string(summary_tokens)
    return summary.strip()


def extract_queries(llm_output: str) -> List[str]:
    if not llm_output:
        return []
    cleaned = llm_output.replace("**", "")
    pattern = r"[-*\u2022]\s*Question\s*\d+\s*:\s*(.+)"
    return [q.strip() for q in re.findall(pattern, cleaned)]


def build_prompt(total_description: str) -> str:
    return f"""
Given the following description of a dataset:

{total_description}

Please identify 5 potential users who would engage with this dataset. For each user, list 5 tasks they would perform with this dataset. Then, for each (user, task) combination, generate 5 questions that require a high-level understanding of the entire dataset.

Output the results in the following structure:
- User 1: [user description]
    - Task 1: [task description]
        - Question 1:
        - Question 2:
        - Question 3:
        - Question 4:
        - Question 5:
    - Task 2: [task description]
    ...
    - Task 5: [task description]
- User 2: [user description]
...
- User 5: [user description]
""".strip()


# -------------------------------
# CLI
# -------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate high-level questions per dataset.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,  # defaults applied in __main__
        help="Directory to write query files. Example: ./working_dir/reduced_datasets/",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,  # defaults applied in __main__
        help="If set, limit each dataset to the first N rows in memory when loading.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,  # defaults applied in __main__
        help="Local cache directory for downloaded datasets.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",      # allow zero or more; defaults applied in __main__
        default=None,
        help="Subset of dataset names to process, e.g.: --datasets cooking history",
    )
    return parser.parse_args()


# -------------------------------
# Main execution
# -------------------------------

def main(
    output_dir: str,
    max_rows: Optional[int],
    save_dir: str,
    dataset_names: List[str],
) -> None:
    logger.info("Starting pipeline.")
    logger.info(f"Configuration -> output_dir={output_dir}, max_rows={max_rows}, save_dir={save_dir}, datasets={dataset_names}")
    logger.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    llm = LLMInvoker()
    logger.info("LLMInvoker initialized.")

    # Ensure output directory exists
    ensure_dir(output_dir)

    # Prepare dataset provider
    provider = UltraDomainDatasetProvider(save_dir=save_dir)
    logger.info("Loading datasets via UltraDomainDatasetProvider ...")
    # NOTE: UltraDomainDatasetProvider.load must support max_rows: Optional[int]
    provider.load(max_rows=max_rows)
    logger.info("Datasets loaded.")

    for name in dataset_names:
        logger.info(f"Processing dataset: {name}")

        contexts = provider.get(name, column="context") or []
        if not contexts:
            logger.warning(f"No contexts found for dataset: {name}. Writing empty file to keep pipeline deterministic.")
            out_path = os.path.join(output_dir, f"{name}_queries.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                pass
            logger.info(f"Saved empty file at: {out_path}")
            continue

        logger.info(f"Building summaries for {name} (total contexts: {len(contexts)})")
        summaries = [get_summary(c, tokenizer=tokenizer) for c in contexts]
        total_description = "\n\n".join(s for s in summaries if s)

        prompt = build_prompt(total_description)

        logger.info(f"Invoking LLM for dataset '{name}' with params: {LLM_PARAMS}")
        try:
            response = llm(parameters=LLM_PARAMS, message=[{"role": "user", "content": prompt}])
        except Exception as e:
            logger.error(f"LLM invocation failed for dataset '{name}': {e}")
            out_path = os.path.join(output_dir, f"{name}_queries.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                pass
            logger.info(f"Saved empty file at: {out_path} due to LLM failure.")
            continue

        preview = total_description[:200].replace("\n", " ")
        logger.info(f"Description preview ({name}): {preview}")
        logger.info(f"Description length (chars): {len(total_description)}")

        truncated = str(response)
        if len(truncated) > 500:
            truncated = truncated[:500] + "..."
        logger.info(f"LLM response (truncated): {truncated}")

        queries = extract_queries(str(response))
        logger.info(f"Extracted {len(queries)} queries for dataset '{name}'.")

        filename = f"{name}_queries.txt"
        out_path = os.path.join(output_dir, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            for q in queries:
                f.write(q + "\n")

        logger.info(f"Saved queries to: {out_path} (total queries: {len(queries)})")

    logger.info("Pipeline finished successfully.")


if __name__ == "__main__":
    # Ensure parent directory is on sys.path if needed
    sys.path.append(os.path.abspath(".."))

    # Defaults (defined ONLY here)
    DEFAULT_OUTPUT_DIR = "./working_dir/reduced_datasets/queries"
    DEFAULT_DATASET_NAMES = ["cooking", "agriculture", "history"]
    DEFAULT_SAVE_DIR = "./working_dir/reduced_datasets/data"
    DEFAULT_MAX_ROWS = 3  # None = load all rows

    # Parse CLI arguments (which may be None)
    args = parse_args()

    # Apply defaults if CLI args are not provided
    main(
        output_dir=args.output_dir or DEFAULT_OUTPUT_DIR,
        max_rows=args.max_rows if args.max_rows is not None else DEFAULT_MAX_ROWS,
        save_dir=args.save_dir or DEFAULT_SAVE_DIR,
        dataset_names=args.datasets or DEFAULT_DATASET_NAMES,
    )

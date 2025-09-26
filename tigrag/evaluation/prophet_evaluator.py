import os
import sys
import json
import logging
import re
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

from tigrag import TemporalInfluenceGraph, TigParam
from tigrag.dataset_provider.prophet_dataset_provider import ProphetDatasetProvider

# run with python3 -m tigrag.evaluation.prophet_evaluator
# ------------------------------
# Hard-coded configuration
# ------------------------------
CONFIG = {
    "workers": 10,
    "workdir": "./working_dir/prophet/tig",
    "llm_name": "claude3_5_bedrock",
    "embed_name": "local",
    "keyword_method": "none",

    "datadir": "./working_dir/prophet/data",
    "load_limit": 100,      # max rows to load from provider
    "start_idx": 0,       # inclusive
    "end_idx": None,      # exclusive; None = until end of df
    "reset_per_row": True,# True: new TIG per row; False: accumulate across rows

    "result_dir": "./working_dir/prophet/results",
    "result_filename": "prophet_answers_3_5.json",
}


def build_corpus_from_articles(articles: Any) -> str:
    if not isinstance(articles, list):
        return ""
    parts: List[str] = []
    for art in articles:
        try:
            title = (art.get("title", "") or "").strip()
            text = (art.get("text", "") or "").strip()
            piece = ""
            if title:
                piece += title + "\n\n"
            piece += text
            if piece.strip():
                parts.append(piece)
        except Exception as e:
            logging.warning(f"Skipping malformed article: {e}")
    return "\n\n---\n\n".join(parts)


def to_native(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    if isinstance(o, (list, tuple, set)):
        return [to_native(x) for x in o]
    if isinstance(o, dict):
        return {str(k): to_native(v) for k, v in o.items()}
    return o


# Zahl am Ende wie "*0.35*" extrahieren
STAR_NUMBER_PATTERN = re.compile(r"\*([+-]?\d+(?:[.,]\d+)?)\*\s*$")

def extract_star_number(text: Any) -> Optional[float]:
    if not isinstance(text, str):
        return None
    m = STAR_NUMBER_PATTERN.search(text.strip())
    if not m:
        return None
    num_str = m.group(1).replace(",", ".")
    try:
        return float(num_str)
    except ValueError:
        return None


def coerce_gt_to01(gt: Any) -> Optional[int]:
    if gt is None:
        return None
    if isinstance(gt, (np.integer, np.floating, np.bool_)):
        val = float(gt)
    elif isinstance(gt, (int, float, bool)):
        val = float(gt)
    elif isinstance(gt, str):
        s = gt.strip()
        if s == "":
            return None
        try:
            val = float(s)
        except ValueError:
            return None
    else:
        return None
    if val in (0.0, 1.0):
        return int(val)
    if abs(val - 0.0) < 1e-12:
        return 0
    if abs(val - 1.0) < 1e-12:
        return 1
    return None


def brier_score(y01: Optional[int], p: Optional[float]) -> Optional[float]:
    if y01 is None or p is None:
        return None
    try:
        p_clipped = min(1.0, max(0.0, float(p)))
        return (p_clipped - float(y01)) ** 2
    except Exception:
        return None


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    params = TigParam(
        embedding_model_name=CONFIG["embed_name"],
        llm_name=CONFIG["llm_name"],
        working_dir=CONFIG["workdir"],
        llm_worker_nodes=CONFIG["workers"],
        keyword_extraction_method=CONFIG["keyword_method"],
    )

    provider = ProphetDatasetProvider(save_dir=CONFIG["datadir"])
    provider.load(CONFIG["load_limit"])
    df = provider.get("prophet_L1")

    if df is None or len(df) == 0:
        logging.error("No data loaded from ProphetDatasetProvider.")
        sys.exit(1)

    n_rows = len(df)
    start = max(0, CONFIG["start_idx"])
    end = n_rows if CONFIG["end_idx"] is None else min(CONFIG["end_idx"], n_rows)

    logging.info(f"Processing rows [{start}:{end}) out of {n_rows} rows.")

    results: List[Dict[str, Any]] = []
    brier_values: List[float] = []

    tig_global = None
    if not CONFIG["reset_per_row"]:
        tig_global = TemporalInfluenceGraph(query_param=params)
        logging.info("Initialized a single TIG index for all rows (accumulating insertions).")

    for i in range(start, end):
        row = df.iloc[i]
        q_text = row["question"] if "question" in df.columns else ""
        gt_answer = row["answer"] if "answer" in df.columns else None

        articles = row["articles"] if "articles" in df.columns else []
        corpus = build_corpus_from_articles(articles)

        tig = tig_global if tig_global is not None else TemporalInfluenceGraph(query_param=params)

        if corpus.strip():
            tig.insert(corpus)
        else:
            logging.warning(f"Row {i}: empty corpus (no articles).")

        if isinstance(q_text, str) and q_text.strip():
            _ = tig.retrieve(q_text.strip())
            model_answer = tig.predict(q_text.strip())
        else:
            model_answer = ""
            logging.warning(f"Row {i}: no question found; skipping retrieval.")

        model_score = extract_star_number(model_answer)
        gt01 = coerce_gt_to01(gt_answer)
        brier = brier_score(gt01, model_score)

        if brier is not None:
            brier_values.append(brier)

        rec: Dict[str, Any] = {
            "question": str(q_text),
            "ground_truth": to_native(gt_answer),
            "model_answer": str(model_answer),
            "model_score": to_native(model_score),
            "brier": to_native(brier),
        }
        results.append(rec)

        if CONFIG["reset_per_row"] and tig_global is None:
            del tig

    # Gesamt-Brier berechnen
    overall_brier = float(np.mean(brier_values)) if brier_values else None

    # Save results + overall_brier in einer JSON-Struktur
    final_output = {
        "results": to_native(results),
        "overall_brier": to_native(overall_brier),
    }

    os.makedirs(CONFIG["result_dir"], exist_ok=True)
    out_path = os.path.join(CONFIG["result_dir"], CONFIG["result_filename"])
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    logging.info(f"Done. Results written to {out_path}")
    if overall_brier is not None:
        logging.info(f"Overall Brier Score: {overall_brier:.4f}")
    else:
        logging.info("No valid Brier scores computed.")

import numpy as np

def calculate_brier_score(predictions, ground_truth):
    """
    Berechnet den Brier Score für binäre Klassifikationen.

    Args:
        predictions: Liste von Wahrscheinlichkeiten (zwischen 0 und 1)
        ground_truth: Liste von tatsächlichen Werten (0 oder 1)

    Returns:
        Brier Score (niedrigere Werte sind besser)
    """
    return np.mean((np.array(predictions) - np.array(ground_truth)) ** 2)


def test_me():
    # Pfad zur JSON-Ergebnisdatei
    result_path = os.path.join(CONFIG["result_dir"], CONFIG["result_filename"])

    # JSON-Datei laden
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Fehler beim Laden der JSON-Datei: {e}")
        return

    predictions = []
    ground_truths = []

    results = data.get("results", [])
    for r in results:
        m_score = r['model_score']
        g_truth = r['ground_truth']
        if not m_score or not g_truth: continue

        # Werte zu den Listen hinzufügen
        predictions.append(m_score)
        ground_truths.append(g_truth)

    # Brier Score berechnen
    if predictions and ground_truths:
        brier_score = calculate_brier_score(predictions, ground_truths)
        print(f"Brier Score: {brier_score:.4f}")
    else:
        print("Keine gültigen Daten für die Berechnung des Brier Scores gefunden.")


if __name__ == "__main__":
    #main()
    test_me()

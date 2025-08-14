# lightrag_query_only.py
# ------------------------------------------------------------
# Lightweight LightRAG client for an ALREADY-BUILT graph.
# - No ingestion here (assumes your LightRAG working_dir already contains data).
# - ask_with_existing_graph(working_dir, query, mode="hybrid")
# - run_queries_file_with_existing_graph(working_dir, queries_file, mode="hybrid", output_json_path=...)
#   -> reads an explicit queries file (one query per line) and writes answers to a JSON file:
#      [
#        {"query": "...", "answer": "..."},
#        {"query": "...", "answer": "..."},
#        ...
#      ]
#
# Requirements:
#   pip install lightrag boto3 torch sentence-transformers tqdm
#
# AWS:
#   export AWS_REGION=eu-west-1
#   export BEDROCK_CLAUDE_MODEL_ID=eu.anthropic.claude-3-7-sonnet-20250219-v1:0
#   (and valid AWS credentials)
# ------------------------------------------------------------
import os
import json
import time
import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Dict
from threading import Lock

import boto3
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger, EmbeddingFunc

# -------------------------------
# Logging & workdir
# -------------------------------
setup_logger("lightrag", level="INFO")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("lightrag_query_only")

DEFAULT_WORKDIR = os.getenv("LIGHTRAG_WORKDIR", "./working_dir/existing_graph")
os.makedirs(DEFAULT_WORKDIR, exist_ok=True)

# Dedicated LLM log
llm_logger = logging.getLogger("llm_calls")
llm_logger.setLevel(logging.INFO)
_llm_log_path = os.path.join(DEFAULT_WORKDIR, "llm_calls.log")
if not any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(_llm_log_path)
           for h in llm_logger.handlers):
    fh = logging.FileHandler(_llm_log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    llm_logger.addHandler(fh)

_LLM_CALL_COUNT = 0
_LLM_LOCK = Lock()

# -------------------------------
# Embeddings (local, MiniLM only)
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
except Exception:
    gpu_name = "CPU"
logger.info(f"Loading embeddings on {device.upper()} ({gpu_name})")
_embed_model = SentenceTransformer("paraphrase-MiniLM-L6-v2", device=device)

async def local_embed_async(texts: List[str]) -> List[List[float]]:
    def _encode(batch: List[str]) -> List[List[float]]:
        return _embed_model.encode(batch, convert_to_numpy=True).tolist()
    return await asyncio.to_thread(_encode, texts)

# -------------------------------
# Bedrock Claude (LLM)
# -------------------------------
BEDROCK_REGION = os.getenv("AWS_REGION", "eu-west-1")
CLAUDE_MODEL_ID = os.getenv(
    "BEDROCK_CLAUDE_MODEL_ID",
    "eu.anthropic.claude-3-7-sonnet-20250219-v1:0",
)
_bedrock = boto3.client(service_name="bedrock-runtime", region_name=BEDROCK_REGION)

def _to_bedrock_messages(history: Optional[List[Dict[str, str]]],
                         user_prompt: str,
                         system_prompt: Optional[str]) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    for m in history or []:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role not in ("user", "assistant", "system"):
            role = "user"
        msgs.append({"role": role, "content": content})
    msgs.append({"role": "user", "content": user_prompt})
    return msgs

async def bedrock_claude_complete(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[List[Dict[str, str]]] = None,
    **parameters
) -> str:
    global _LLM_CALL_COUNT
    with _LLM_LOCK:
        _LLM_CALL_COUNT += 1
        call_id = _LLM_CALL_COUNT

    start = time.perf_counter()

    def _invoke() -> str:
        max_new_tokens = parameters.get("max_new_tokens", 4096)
        temperature = parameters.get("temperature", 0.7)
        top_p = parameters.get("top_p", 0.95)

        messages = _to_bedrock_messages(history_messages, prompt, system_prompt)
        system_val = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        message_body = messages[1:] if system_val else messages

        in_chars = sum(len(m.get("content", "")) for m in message_body) + len(system_val)
        llm_logger.info(
            f"[#{call_id}] START model={CLAUDE_MODEL_ID} region={BEDROCK_REGION} "
            f"temp={temperature} top_p={top_p} max_tokens={max_new_tokens} "
            f"history={len(history_messages or [])} in_chars={in_chars}"
        )

        body = json.dumps({
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "system": system_val,
            "messages": message_body,
            "anthropic_version": "bedrock-2023-05-31",
        })

        resp = _bedrock.invoke_model(
            body=body,
            modelId=CLAUDE_MODEL_ID,
            accept="application/json",
            contentType="application/json",
        )
        payload = json.loads(resp["body"].read())
        parts = payload.get("content", [])
        text = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
        return text.strip()

    try:
        out = await asyncio.to_thread(_invoke)
        dur = time.perf_counter() - start
        llm_logger.info(f"[#{call_id}] END   ok duration_sec={dur:.3f} out_chars={len(out)}")
        return out
    except Exception as e:
        dur = time.perf_counter() - start
        llm_logger.exception(f"[#{call_id}] END   error duration_sec={dur:.3f} reason={e}")
        raise

# -------------------------------
# Utility: load queries from explicit file path
# -------------------------------
def load_queries_from_file(file_path: Path) -> List[str]:
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"Queries file not found: {file_path}")
    lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    # Skip empty lines & comments (# ...)
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]

# -------------------------------
# Core asyncio helpers (no ingestion)
# -------------------------------
async def _init_rag(working_dir: str | Path) -> LightRAG:
    rag = LightRAG(
        working_dir=str(working_dir),
        llm_model_max_async=10,
        llm_model_func=bedrock_claude_complete,
        embedding_func=EmbeddingFunc(
            embedding_dim=384,   # MiniLM-L6-v2
            max_token_size=512,
            func=local_embed_async,
        ),
    )
    # Initialize storages only; do NOT insert anything here.
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag

async def _aquery(rag: LightRAG, query: str, mode: str = "hybrid") -> str:
    return await rag.aquery(query, param=QueryParam(mode=mode))

async def _close_rag(rag: Optional[LightRAG]) -> None:
    if rag:
        await rag.finalize_storages()
    global _LLM_CALL_COUNT
    logger.info("Total number of LLM calls: #%d", _LLM_CALL_COUNT)

# -------------------------------
# Public sync functions
# -------------------------------
def ask_with_existing_graph(working_dir: str | Path, query: str, mode: str = "hybrid") -> str:
    """
    Load an existing LightRAG graph from `working_dir` and ask a single query.
    """
    async def _run() -> str:
        rag = await _init_rag(working_dir)
        try:
            return await _aquery(rag, query, mode=mode)
        finally:
            await _close_rag(rag)
    return asyncio.run(_run())

def run_queries_file_with_existing_graph(
    working_dir: str | Path,
    queries_file: str | Path,
    mode: str = "hybrid",
    output_json_path: Optional[str | Path] = None,
) -> List[Dict[str, str]]:
    """
    Load an existing LightRAG graph from `working_dir` and iterate queries from
    an explicit file path `queries_file` (one query per line).
    If `output_json_path` is provided, write the results as a JSON array to that path:
      [
        {"query": "...", "answer": "..."},
        {"query": "...", "answer": "..."}
      ]
    If `output_json_path` is a directory or has no .json suffix, 'answers.json' is created inside it.
    """
    async def _run() -> List[Dict[str, str]]:
        rag = await _init_rag(working_dir)
        try:
            qfile = Path(queries_file)
            queries = load_queries_from_file(qfile)
            results: List[Dict[str, str]] = []
            for q in tqdm(queries, desc="Query LightRAG"):
                ans = await _aquery(rag, q, mode=mode)
                results.append({"query": q, "answer": ans})

            if output_json_path:
                outp = Path(output_json_path)
                # If it's a directory or missing suffix, default to 'answers.json'
                if outp.is_dir() or outp.suffix.lower() != ".json":
                    outp = outp / "answers.json"
                outp.parent.mkdir(parents=True, exist_ok=True)
                with outp.open("w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                logger.info("Results saved to: %s", outp)

            return results
        finally:
            await _close_rag(rag)
    return asyncio.run(_run())

# -------------------------------
# Simple manual runner (no argparse)
# -------------------------------
if __name__ == "__main__":
    # Configure your existing LightRAG working dir (already populated elsewhere)

    # run with python3 -m tigrag.evaluation.lightrag.run_lightrag_query

    WORKDIR = "./working_dir/reduced_datasets/lightrag/cooking_dataset"

    # Choose run type: "ask" (single query) or "runfile" (iterate queries from an explicit FILE)
    RUN = "runfile"  # or: "ask"
    MODE = "hybrid" # hybrid or naive

    if RUN == "ask":
        QUERY = "How do the acknowledgments reflect the collaborative nature of culinary education between academic institutions and industry professionals?"
        answer = ask_with_existing_graph(
            working_dir=WORKDIR,
            query=QUERY,
            mode=MODE,
        )
        print("\n=== ANSWER ===\n")
        print(answer)

    elif RUN == "runfile":
        # Explicit path to your queries file (one query per line)
        QUERIES_FILE = "./working_dir/reduced_datasets/queries/cooking_queries.txt"

        # Output path for JSON (either a file like ".../my_answers.json" OR a directory;
        # if it's a directory or has no .json suffix, 'answers.json' will be created in it)
        OUTPUT_JSON = f"./working_dir/reduced_datasets/answers/{MODE}_cooking_answers.json"

        results = run_queries_file_with_existing_graph(
            working_dir=WORKDIR,
            queries_file=QUERIES_FILE,
            mode=MODE,
            output_json_path=OUTPUT_JSON,
        )
        print(f"\n=== DONE – {len(results)} answers ===")
        for i, item in enumerate(results[:3], 1):
            print(f"\n[{i}] Q: {item['query']}\nA: {item['answer'][:500]}{'…' if len(item['answer'])>500 else ''}")

    else:
        raise SystemExit("Unknown RUN value. Expected 'ask' or 'runfile'.")

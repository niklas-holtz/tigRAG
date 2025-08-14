# run_lightrag.py
import os
import json
import time
import asyncio
import logging
from typing import List, Dict, Any, Optional
from threading import Lock

# dataset
from tigrag.dataset_provider.ultradomain_dataset_provider import UltraDomainDatasetProvider

import boto3
import torch
from sentence_transformers import SentenceTransformer

from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger, EmbeddingFunc


# -------------------------------
# Logging and workspace
# -------------------------------
setup_logger("lightrag", level="INFO")
logging.basicConfig(level=logging.INFO)

WORKING_DIR = os.getenv("LIGHTRAG_WORKDIR", "./working_dir/reduced_datasets/lightrag/cooking_dataset")
os.makedirs(WORKING_DIR, exist_ok=True)

# Dedicated LLM call logger -> ./rag_storage/llm_calls.log
llm_logger = logging.getLogger("llm_calls")
llm_logger.setLevel(logging.INFO)
_llm_log_path = os.path.join(WORKING_DIR, "llm_calls.log")
if not any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(_llm_log_path)
           for h in llm_logger.handlers):
    fh = logging.FileHandler(_llm_log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    llm_logger.addHandler(fh)

# Global LLM call counter (thread-safe)
_LLM_CALL_COUNT = 0
_LLM_LOCK = Lock()


# -------------------------------
# Local embeddings (MiniLM-L6-v2)
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(
    f"Loading embedding model on {device.upper()} "
    f"({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})"
)
_embed_model = SentenceTransformer("paraphrase-MiniLM-L6-v2", device=device)

async def local_embed_async(texts: List[str]) -> List[List[float]]:
    """Async wrapper around SentenceTransformer.encode."""
    def _encode(batch: List[str]) -> List[List[float]]:
        return _embed_model.encode(batch, convert_to_numpy=True).tolist()
    return await asyncio.to_thread(_encode, texts)


# -------------------------------
# Bedrock Claude (same style as your invoker)
# -------------------------------
BEDROCK_REGION = os.getenv("AWS_REGION", "eu-west-1")
CLAUDE_MODEL_ID = os.getenv(
    "BEDROCK_CLAUDE_MODEL_ID",
    "eu.anthropic.claude-3-7-sonnet-20250219-v1:0"
)
_bedrock = boto3.client(service_name="bedrock-runtime", region_name=BEDROCK_REGION)

def _to_bedrock_messages(
    history_messages: Optional[List[Dict[str, str]]],
    user_prompt: str,
    system_prompt: Optional[str]
) -> List[Dict[str, str]]:
    """Map LightRAG (system, history, user) to Bedrock/Anthropic chat format."""
    msgs: List[Dict[str, str]] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})

    for m in history_messages or []:
        if isinstance(m, dict):
            role = m.get("role", "user")
            content = m.get("content", "")
        else:
            role, content = m
        if role not in ("user", "assistant", "system"):
            role = "user"
        msgs.append({"role": role, "content": content})

    msgs.append({"role": "user", "content": user_prompt})
    return msgs

async def bedrock_claude_complete(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[List[Dict[str, str]]] = None,
    **parameters: Any
) -> str:
    """
    LightRAG LLM function using Bedrock Claude with exactly your request structure.
    Also increments a global call counter and logs details per call.
    """
    # Reserve a call_id atomically
    global _LLM_CALL_COUNT
    with _LLM_LOCK:
        _LLM_CALL_COUNT += 1
        call_id = _LLM_CALL_COUNT

    start = time.perf_counter()

    def _invoke() -> str:
        max_new_tokens = parameters.get("max_new_tokens", 6024)
        temperature = parameters.get("temperature", 0.8)
        top_p = parameters.get("top_p", 0.95)

        messages = _to_bedrock_messages(history_messages, prompt, system_prompt)
        system_val = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        message_body = messages[1:] if system_val else messages

        # Pre-log basic input stats
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
            "anthropic_version": "bedrock-2023-05-31"
        })

        resp = _bedrock.invoke_model(
            body=body,
            modelId=CLAUDE_MODEL_ID,
            accept="application/json",
            contentType="application/json"
        )
        payload = json.loads(resp["body"].read())
        parts = payload.get("content", [])
        text = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
        return text.strip()

    try:
        out = await asyncio.to_thread(_invoke)
        duration = time.perf_counter() - start
        llm_logger.info(f"[#{call_id}] END   ok duration_sec={duration:.3f} out_chars={len(out)}")
        return out
    except Exception as e:
        duration = time.perf_counter() - start
        llm_logger.exception(f"[#{call_id}] END   error duration_sec={duration:.3f} reason={e}")
        raise


# -------------------------------
# LightRAG initialization
# -------------------------------
async def initialize_rag() -> LightRAG:
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_max_async=10,
        llm_model_func=bedrock_claude_complete,
        embedding_func=EmbeddingFunc(
            embedding_dim=384,     # MiniLM-L6-v2
            max_token_size=512,    # safe input length for MiniLM
            func=local_embed_async
        ),
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag


# -------------------------------
# Example main
# -------------------------------
async def main() -> None:
    rag: Optional[LightRAG] = None
    try:
        rag = await initialize_rag()
        
        provider = UltraDomainDatasetProvider(save_dir='./working_dir/reduced_datasets/data')
        provider.load(3)
        cooking_text = ' '.join(provider.get("cooking", column="context"))

        await rag.ainsert(provider.get("cooking", column="context"))
        #for d in docs:
        #    await rag.ainsert(d)

        answer = await rag.aquery(
            "How do traditional cooking methods compare with modern approaches in the various texts?",
            param=QueryParam(mode="hybrid")
        )
        print("\n=== ANSWER ===\n", answer)

    except Exception as e:
        logging.error(f"Run failed: {e}", exc_info=True)
    finally:
        # Report total LLM calls
        global _LLM_CALL_COUNT
        logging.info(f"Total LLM calls in this run: #{_LLM_CALL_COUNT}")
        if rag:
            await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())

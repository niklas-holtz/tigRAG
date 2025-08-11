from __future__ import annotations
import logging
import json
import re
from typing import Any, Dict, List, Optional, Callable, Tuple
from ..steps.step import Step, RetrieveContext
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from string import Template
from datetime import datetime

class EventExtractorStep(Step):
    """
    Extracts structured events from selected chunks using an LLM prompt in parallel.
    - Iterates over ctx.selected (list of chunks)
    - For each chunk, builds a prompt and calls the provided LLM
    - Parses the response as JSON (robustly)
    - Stores a list of event dicts into ctx.events
    """

    def __init__(
        self,
        max_retries: int = 2,
        max_workers: int = 1
    ):
        self.prompt = self._load_prompt('../prompts/influence_event_prompt.txt')
        self._prompt_tmpl = Template(self.prompt) if self.prompt else None
        self.max_retries = int(max_retries)
        self.max_workers = int(max_workers)

    def _load_prompt(self, path: str) -> str:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        abs_path = os.path.join(base_dir, path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Prompt file not found: {abs_path}")
        with open(abs_path, "r", encoding="utf-8") as f:
            return f.read()

    def run(self, ctx: RetrieveContext) -> RetrieveContext:
        logging.info("Starting EventExtractorStep...")

        if not getattr(ctx, "selected", None):
            logging.warning("No selected chunks found in context — nothing to extract.")
            ctx.events = []
            return ctx

        query_text = getattr(ctx, "query", None)

        # Prepare tasks for all chunks
        tasks: List[Tuple[int, Dict[str, Any], str]] = []
        for i, chunk in enumerate(ctx.selected[:2], start=1):
            prompt = self._build_prompt(chunk.get("text", "") or "", query_text)
            tasks.append((i, chunk, prompt))

        events: List[Dict[str, Any]] = []

        # Process in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._process_chunk, idx, chunk, prompt, ctx): (idx, chunk)
                for idx, chunk, prompt in tasks
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Extracting events (parallel)",
                unit="chunk"
            ):
                idx, chunk = futures[future]
                try:
                    parsed_event = future.result()
                except Exception as e:
                    logging.error(f"Error extracting event for chunk #{idx}: {e}")
                    parsed_event = {
                        "event_type": None,
                        "entities": [],
                        "time": None,
                        "location": None,
                        "summary": None,
                        "confidence": 0.0,
                        "_error": str(e),
                    }
                    self._attach_metadata(parsed_event, chunk)
                events.append(parsed_event)

        ctx.events = events

        try:
            if hasattr(ctx, "chunk_storage") and ctx.chunk_storage is not None:
                retrieved_at = datetime.utcnow().isoformat()
                ctx.chunk_storage.insert_events(
                    query=query_text or "",
                    events=events,
                    retrieved_at=retrieved_at
                )
                logging.info(f"Saved {len(events)} events for query '{query_text}' into database.")
            else:
                logging.warning("No chunk_storage found in context — events not persisted.")
        except Exception as e:
            logging.error(f"Failed to save events to database: {e}")

        logging.info(f"EventExtractorStep finished. Extracted {len(events)} event(s).")
        return ctx

    # ------------------------ helpers ------------------------

    def _process_chunk(self, idx: int, chunk: Dict[str, Any], prompt: str, ctx: RetrieveContext) -> Dict[str, Any]:
        """Runs the LLM for one chunk, retries on parse errors, returns parsed event dict."""
        raw = None
        parsed: Optional[Dict[str, Any]] = None

        for attempt in range(1, self.max_retries + 2):
            try:
                raw = ctx.llm_invoker(messages=[{"role": "user", "content": prompt}], max_new_tokens=4000)
                logging.info(f'LLM response: {raw}')
                parsed = self._parse_first_json_object(raw)
                if parsed is not None:
                    break
            except Exception as e:
                logging.warning(f"LLM call failed for chunk #{idx} (attempt {attempt}): {e}")

        if parsed is None:
            logging.warning(f"Could not parse JSON for chunk #{idx}.")
            parsed = {
                "event_type": None,
                "entities": [],
                "time": None,
                "location": None,
                "summary": None,
                "confidence": 0.0,
                "_raw": raw,
            }

        self._attach_metadata(parsed, chunk)
        return parsed

    def _attach_metadata(self, event: Dict[str, Any], chunk: Dict[str, Any]) -> None:
        """Attach source metadata to the parsed event."""
        event["_chunk_id"] = chunk.get("id")
        event["_cluster_id"] = chunk.get("cluster_id")
        event["_source_from_idx"] = chunk.get("from_idx")
        event["_source_to_idx"] = chunk.get("to_idx")

    def _build_prompt(self, chunk_text: str, query_text: Optional[str]) -> str:
        if not self._prompt_tmpl:
            raise ValueError("No prompt template loaded.")
        return self._prompt_tmpl.safe_substitute(
            query=(query_text or ""),
            chunk_text=chunk_text
        )

    @staticmethod
    def _parse_first_json_object(text: Optional[str]) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        start_positions = [m.start() for m in re.finditer(r"\{", text)]
        for start in start_positions:
            depth = 0
            for end in range(start, len(text)):
                if text[end] == "{":
                    depth += 1
                elif text[end] == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : end + 1]
                        try:
                            obj = json.loads(candidate)
                            if isinstance(obj, dict):
                                return obj
                        except Exception:
                            break
        return None

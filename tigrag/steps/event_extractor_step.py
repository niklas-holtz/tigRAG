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
    Extracts structured events (array of event objects) from selected chunks using an LLM in parallel.
    The LLM is expected to return a strict JSON array:
    [
      {
        "title": "...",
        "clues": ["...", "..."],
        "associated_numbers": [ { "value": "...", "explanation": "..." } ],
        "actors": ["...", "..."],
        "locations": ["US", "EU"],
        "summary": "...",
        "influence": ["...", "..."],
        "confidence": 0.0
      }
    ]
    If no relevant event: []
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
        self.max_workers = getattr(ctx, "llm_worker_nodes", 1)

        # Prepare tasks for all chunks (nimm hier ggf. [:2] raus, falls du alles willst)
        tasks: List[Tuple[int, Dict[str, Any], str]] = []
        for i, chunk in enumerate(ctx.selected, start=1):
            prompt = self._build_prompt(chunk.get("text", "") or "", query_text)
            tasks.append((i, chunk, prompt))

        all_events: List[Dict[str, Any]] = []

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
                    events_for_chunk = future.result()  # List[Dict]
                except Exception as e:
                    logging.error(f"Error extracting event for chunk #{idx}: {e}")
                    events_for_chunk = []

                # attach metadata per event + collect
                for ev in events_for_chunk:
                    self._attach_metadata(ev, chunk)
                all_events.extend(events_for_chunk)

        # Allocate ids
        for idx, ev in enumerate(all_events, start=1):
            ev["id"] = idx

        ctx.events = all_events

        # Persist to DB
        try:
            if hasattr(ctx, "chunk_storage") and ctx.chunk_storage is not None:
                retrieved_at = datetime.utcnow().isoformat()
                inserted_row_id = ctx.chunk_storage.insert_events(
                    query=query_text or "",
                    events=all_events,
                    retrieved_at=retrieved_at
                )
                ctx.event_row_id = inserted_row_id
                logging.info(f"Saved {len(all_events)} events for query '{query_text}' into database.")
            else:
                logging.warning("No chunk_storage found in context — events not persisted.")
        except Exception as e:
            logging.error(f"Failed to save events to database: {e}")

        logging.info(f"EventExtractorStep finished. Extracted {len(all_events)} event(s).")
        return ctx

    def _process_chunk(self, idx: int, chunk: Dict[str, Any], prompt: str, ctx: RetrieveContext) -> List[
        Dict[str, Any]]:
        raw = None
        events_list: Optional[List[Dict[str, Any]]] = None
        thoughts_text: Optional[str] = None

        for attempt in range(1, self.max_retries + 2):
            try:
                param = {
                    'max_new_tokens': 3000
                }
                raw = ctx.llm_invoker(message=[{"role": "user", "content": prompt}], parameters=param)
                logging.info(
                    f'LLM response (chunk #{idx}, attempt {attempt}): {raw[:500]}{"..." if len(str(raw)) > 500 else ""}')
                thoughts_text, json_text = self._split_thoughts_and_json(raw)

                # Handle 'None'
                if json_text is not None and json_text.strip().lower() == "none":
                    events_list = []
                    break

                # Parse array (or object with "events" already normalized by splitter)
                if json_text:
                    try:
                        obj = json.loads(json_text)
                        if isinstance(obj, list):
                            events_list = obj
                            break
                        if isinstance(obj, dict) and "events" in obj and isinstance(obj["events"], list):
                            events_list = obj["events"]
                            break
                    except Exception as e:
                        logging.warning(f"JSON parse failed for chunk #{idx} (attempt {attempt}): {e}")
            except Exception as e:
                logging.warning(f"LLM call failed for chunk #{idx} (attempt {attempt}): {e}")

        if not isinstance(events_list, list):
            logging.warning(f"Could not parse JSON array for chunk #{idx}. Returning empty list.")
            events_list = []

        # Optional: Thoughts als Debug-Meta an JEDEM Event anhängen (oder separat sammeln)
        clean: List[Dict[str, Any]] = []
        for ev in events_list:
            if isinstance(ev, dict):
                if thoughts_text:
                    ev["_thoughts"] = thoughts_text  # optional; zum Debuggen
                clean.append(ev)
        return clean

    def _attach_metadata(self, event: Dict[str, Any], chunk: Dict[str, Any]) -> None:
        """Attach source metadata to a single event object."""
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
    def _parse_first_json_array_or_events(text: Optional[str]) -> Optional[List[Dict[str, Any]]]:
        """
        Try to parse:
        - a JSON array directly: [ {...}, {...} ]
        - or an object with key "events": { "events": [ ... ] }
        - or the literal 'None' -> []
        Robustly extracts the first [...] block if needed.
        """
        if text is None:
            return None

        s = text.strip()
        if s.lower() == "none":
            return []
        # direct parse
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict) and "events" in obj and isinstance(obj["events"], list):
                return obj["events"]
        except Exception:
            pass

        # find first [...] bracketed array
        start_positions = [m.start() for m in re.finditer(r"\[", s)]
        for start in start_positions:
            depth = 0
            for end in range(start, len(s)):
                if s[end] == "[":
                    depth += 1
                elif s[end] == "]":
                    depth -= 1
                    if depth == 0:
                        candidate = s[start : end + 1]
                        try:
                            arr = json.loads(candidate)
                            if isinstance(arr, list):
                                return arr
                        except Exception:
                            break  # try next start
        return None

    def _split_thoughts_and_json(self, text: Optional[str]) -> tuple[Optional[str], Optional[str]]:
        """
        Returns (thoughts_text, json_text). If no explicit sections are found,
        tries to heuristically find the first JSON array.
        """
        if not text:
            return None, None

        s = text.strip()
        # Hard rule: literal None -> no events
        if s.lower() == "none":
            return s, "[]"

        # Look for explicit markers
        # e.g.
        # **THOUGHTS**
        # ...free text...
        # **JSON**
        # [ ... ]
        thoughts, json_part = None, None

        # Normalize markers (allow variations)
        thoughts_marker = re.search(r"\*\*THOUGHTS\*\*", s, flags=re.IGNORECASE)
        json_marker = re.search(r"\*\*JSON\*\*", s, flags=re.IGNORECASE)

        if thoughts_marker and json_marker and thoughts_marker.start() < json_marker.start():
            thoughts = s[thoughts_marker.end():json_marker.start()].strip()
            json_part = s[json_marker.end():].strip()

        # If JSON marker missing, try to extract first JSON array
        if json_part is None:
            # try direct array/object first
            try:
                obj = json.loads(s)
                if isinstance(obj, list):
                    return thoughts, s
                if isinstance(obj, dict) and "events" in obj:
                    return thoughts, json.dumps(obj["events"], ensure_ascii=False)
            except Exception:
                pass

            # fallback: find first [...] block
            start_positions = [m.start() for m in re.finditer(r"\[", s)]
            for start in start_positions:
                depth = 0
                for end in range(start, len(s)):
                    if s[end] == "[":
                        depth += 1
                    elif s[end] == "]":
                        depth -= 1
                        if depth == 0:
                            candidate = s[start:end + 1]
                            # sanity check
                            try:
                                arr = json.loads(candidate)
                                if isinstance(arr, list):
                                    json_part = candidate
                                    break
                            except Exception:
                                break
                if json_part is not None:
                    break

        return thoughts, json_part

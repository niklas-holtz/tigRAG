from __future__ import annotations
import logging
import json
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from string import Template
from ..steps.step import Step, RetrieveContext
from tqdm import tqdm


class AnswerGenerationStep(Step):
    """
    Generates a natural-language answer from events, relations, and ratings, and then calls an LLM
    with a prompt that receives only the user query and the constructed retrieval_context.

    Flow:
    1) Load (events, relations, ratings) from storage (if available) or from ctx.
    2) Rank events by rating.score and select top_n.
    3) Build a retrieval_context text that lists the top events with their fields and relations.
    4) Load a prompt template (from disk, provided via constructor), fill in {query, retrieval_context}.
    5) Invoke ctx.llm_invoker and store the LLM output in ctx.retrieval_answer.

    Notes:
    - Output text for the context is English-only.
    - All comments are in English as requested.
    """

    def __init__(self, top_n: int = 30, prompt_path: str = "../prompts/answer_generation_prompt.txt"):
        self.top_n = int(top_n)
        # Load the prompt template once; will be filled with {query, retrieval_context}
        self.prompt_text: str = self._load_prompt(prompt_path)
        self._prompt_tmpl: Template = Template(self.prompt_text)

    # --- helper: robustly parse relation label from relation dict ---
    @staticmethod
    def _parse_relation_label(rel: Dict[str, Any]) -> Optional[str]:
        # Try direct field
        label = rel.get("relation")
        if isinstance(label, str):
            return label.strip().lower()

        # Try to parse from JSON string/dict inside relation_raw
        raw = rel.get("relation_raw")
        if isinstance(raw, dict):
            lab = raw.get("relation")
            return lab.strip().lower() if isinstance(lab, str) else None
        if isinstance(raw, str):
            try:
                obj = json.loads(raw)
                lab = obj.get("relation")
                return lab.strip().lower() if isinstance(lab, str) else None
            except Exception:
                return None
        return None

    # --- helper: stringify list fields safely ---
    @staticmethod
    def _fmt_list(items: Any) -> str:
        if not items:
            return "-"
        if isinstance(items, list):
            return "; ".join([str(x) for x in items if x is not None and str(x).strip() != ""]) or "-"
        return str(items)

    # --- helper: stringify associated_numbers ---
    @staticmethod
    def _fmt_associated_numbers(nums: Any) -> str:
        if not nums or not isinstance(nums, list):
            return "-"
        parts = []
        for n in nums:
            if isinstance(n, dict):
                val = n.get("value")
                expl = n.get("explanation")
                if val is None and expl is None:
                    continue
                if expl:
                    parts.append(f"{val} ({expl})")
                else:
                    parts.append(f"{val}")
            else:
                parts.append(str(n))
        return "; ".join(parts) if parts else "-"

    # --- loader: prefer DB get_events_full, fallback to ctx ---
    def _load_all(self, ctx: RetrieveContext) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        events: List[Dict[str, Any]] = getattr(ctx, "events", []) or []
        relations: List[Dict[str, Any]] = getattr(ctx, "event_relations", []) or []
        ratings: List[Dict[str, Any]] = getattr(ctx, "event_ratings", []) or []

        storage = getattr(ctx, "chunk_storage", None)
        query = getattr(ctx, "query", None)

        # If storage is available, try to load the most recent triple (events, relations, ratings) for the query
        if storage is not None:
            try:
                _eid, evs, rels, rats = storage.get_events_full(query=query)
                if evs:
                    events = evs
                if rels:
                    relations = rels
                if rats:
                    ratings = rats
            except Exception as e:
                logging.warning(f"Could not load events_full from storage: {e}")

        # Ensure every event has an 'id'
        if events and any(ev.get("id") is None for ev in events):
            logging.warning("Some events missing 'id'. Assigning sequential IDs (1..n).")
            for i, ev in enumerate(events, start=1):
                if ev.get("id") is None:
                    ev["id"] = i

        return events, relations, ratings

    # --- indexing helpers ---
    @staticmethod
    def _index_by_id(events: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        out: Dict[int, Dict[str, Any]] = {}
        for ev in events:
            try:
                eid = int(ev.get("id"))
                out[eid] = ev
            except Exception:
                continue
        return out

    @staticmethod
    def _score_map(ratings: List[Dict[str, Any]]) -> Dict[int, float]:
        smap: Dict[int, float] = {}
        for r in ratings:
            try:
                eid = int(r.get("id"))
                score = float(r.get("score", 0.0))
                smap[eid] = score
            except Exception:
                continue
        return smap

    def run(self, ctx: RetrieveContext) -> RetrieveContext:
        logging.info("Starting AnswerGenerationStep...")

        # 1) Load events, relations, ratings
        events, relations, ratings = self._load_all(ctx)
        if not events:
            logging.warning("No events available for answer generation.")
            ctx.retrieval_context = "No events available."
            ctx.retrieval_answer = "No answer generated because no events were available."
            return ctx

        id_to_event = self._index_by_id(events)
        score_of = self._score_map(ratings)

        # 2) Sort events by score (desc), fallback score=0.0
        def score_key(ev: Dict[str, Any]) -> float:
            try:
                return float(score_of.get(int(ev.get("id")), 0.0))
            except Exception:
                return 0.0

        sorted_events = sorted(events, key=score_key, reverse=True)
        top_events = sorted_events[: self.top_n]

        # 3) Build adjacency lists of relations for quick lookup
        outgoing: Dict[int, List[Dict[str, Any]]] = {}
        incoming: Dict[int, List[Dict[str, Any]]] = {}

        for rel in relations or []:
            src = rel.get("source_id")
            tgt = rel.get("target_id")
            if src is None or tgt is None:
                # Support legacy index fields if needed
                src_idx = rel.get("_source_idx")
                tgt_idx = rel.get("_target_idx")
                try:
                    if src is None and isinstance(src_idx, int) and 0 <= src_idx < len(events):
                        src = events[src_idx].get("id")
                    if tgt is None and isinstance(tgt_idx, int) and 0 <= tgt_idx < len(events):
                        tgt = events[tgt_idx].get("id")
                except Exception:
                    pass

            # Only record edges between known IDs
            try:
                src = int(src) if src is not None else None
                tgt = int(tgt) if tgt is not None else None
            except Exception:
                src, tgt = None, None

            if src is None or tgt is None:
                continue

            label = self._parse_relation_label(rel) or "none"

            # Store minimal relation view
            rel_view = {
                "from": src,
                "to": tgt,
                "relation": label
            }

            outgoing.setdefault(src, []).append(rel_view)
            incoming.setdefault(tgt, []).append(rel_view)

        # 4) Compose the retrieval context (English only)
        lines: List[str] = []
        lines.append("# Top Events Overview\n")
        lines.append(f"Below are the top {len(top_events)} events ranked by their influence score.\n")

        for rank, ev in enumerate(tqdm(top_events, desc="Building context", unit="event"), start=1):
            eid = int(ev.get("id"))
            score = score_of.get(eid, 0.0)

            title = ev.get("title") or f"Event {eid}"
            summary = ev.get("summary") or ev.get("description") or "-"
            actors = self._fmt_list(ev.get("actors"))
            locations = self._fmt_list(ev.get("locations"))
            assoc = self._fmt_associated_numbers(ev.get("associated_numbers"))
            influence = self._fmt_list(ev.get("influence"))

            # Collect compact relation strings
            out_edges = outgoing.get(eid, [])
            in_edges = incoming.get(eid, [])

            out_str = ", ".join([f"({e['from']} -> {e['to']}: {e['relation']})" for e in out_edges]) if out_edges else "-"
            in_str  = ", ".join([f"({e['from']} -> {e['to']}: {e['relation']})" for e in in_edges])  if in_edges  else "-"

            lines.append(f"## {rank}. [{eid}] {title}")
            lines.append(f"**Score:** {score:.4f}")
            lines.append(f"**Description:** {summary}")
            lines.append(f"**Actors:** {actors}")
            lines.append(f"**Locations:** {locations}")
            lines.append(f"**Associated Numbers:** {assoc}")
            lines.append(f"**Influence chain:** {influence}")
            lines.append(f"**Relations (outgoing):** {out_str}")
            lines.append(f"**Relations (incoming):** {in_str}")
            lines.append("")  # blank line between cards

        retrieval_context = "\n".join(lines).strip()
        ctx.retrieval_context = retrieval_context  # keep the assembled context in ctx

        # 5) Build the LLM prompt and invoke the model (only query + retrieval_context)
        try:
            user_query = getattr(ctx, "query", "") or ""
            filled_prompt = self._prompt_tmpl.safe_substitute(
                query=user_query,
                retrieval_context=retrieval_context
            )
        except Exception as e:
            logging.error(f"Failed to build answer-generation prompt: {e}")
            ctx.retrieval_answer = "Failed to build answer-generation prompt."
            return ctx

        try:
            params = {'max_new_tokens': 2000}
            llm_response = ctx.llm_invoker(
                message=[{"role": "user", "content": filled_prompt}],
                parameters=params
            )
            ctx.retrieval_answer = llm_response
        except Exception as e:
            logging.error(f"LLM invocation failed for AnswerGenerationStep: {e}")
            ctx.retrieval_answer = "Answer generation failed due to LLM invocation error."

        logging.info("AnswerGenerationStep finished.")
        return ctx

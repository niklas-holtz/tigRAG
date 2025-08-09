"""
EventRAG-Pipeline with TextChunker and debug prints.
----------------------------------------------------
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import json
import uuid
import math

@dataclass
class Entity:
    name: str
    type: str
    aliases: List[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: f"EN_{uuid.uuid4().hex[:8]}")

@dataclass
class Event:
    title: str
    description: str
    time: Optional[str] = None
    participants: List[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: f"EV_{uuid.uuid4().hex[:8]}")
    embedding: Optional[List[float]] = None

@dataclass
class Relation:
    source: str
    target: str
    type: str
    evidence: Optional[str] = None
    id: str = field(default_factory=lambda: f"RL_{uuid.uuid4().hex[:8]}")

@dataclass
class EventKnowledgeGraph:
    events: Dict[str, Event] = field(default_factory=dict)
    entities: Dict[str, Entity] = field(default_factory=dict)
    relations: List[Relation] = field(default_factory=list)

    def add_event(self, ev: Event):
        self.events[ev.id] = ev

    def add_entity(self, ent: Entity):
        self.entities[ent.id] = ent

    def add_relation(self, rel: Relation):
        self.relations.append(rel)

    def neighbors(self, event_id: str) -> List[Event]:
        out: List[Event] = []
        for r in self.relations:
            if r.source == event_id and r.target in self.events:
                out.append(self.events[r.target])
            if r.target == event_id and r.source in self.events:
                out.append(self.events[r.source])
        seen = set()
        uniq = []
        for e in out:
            if e.id not in seen:
                seen.add(e.id)
                uniq.append(e)
        return uniq

def cosine_sim(a: Optional[List[float]], b: Optional[List[float]]) -> float:
    if not a or not b:
        return -1.0
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a)) or 1e-12
    nb = math.sqrt(sum(y*y for y in b)) or 1e-12
    return dot / (na * nb)

EXTRACTION_SYSTEM = (
    "You extract structured event data for an Event Knowledge Graph. "
    "Follow the schema EXACTLY. Output ONE JSON object only. "
    "No explanations, no markdown, no prose."
)

EXTRACTION_USER_TMPL = """Read the text and return EXACTLY ONE JSON object with keys: events, entities, relations.

Schema:
- events: list of objects
  - title: string (concise)
  - description: string (what happened)
  - time: string or null (e.g., '2023-03-15' or 'March 15, 2023'); omit if unknown
  - participants: list of strings (entity names from 'entities')
- entities: list of objects
  - name: string
  - type: one of ["Person","Organisation","Location","Other"]
  - aliases: list of strings (optional; omit if none)
- relations: list of objects
  - from_title: string (must match an 'events[].title')
  - to_title: string (must match an 'events[].title')
  - type: one of ["precedes","follows","causes","related_to","similar"]

Strict rules:
- Output ONE and ONLY ONE top-level JSON object.
- Do NOT repeat the object. Do NOT append a second object.
- Do NOT include fields not defined in the schema.
- Put relation fields ONLY in 'relations', NOT inside 'events'.
- 'participants' MUST be names present in 'entities'.
- If uncertain about a value, omit the field or use null.
- Keep values concise. No markdown, no code fences, no commentary.

Text:
<<<
{chunk}
>>>

Return ONLY the JSON object. Start with '{{' and end with '}}'.
"""


AGENT_SYSTEM = (
    "You are an EventRAG reasoning agent. "
    "You receive a user question and a list of relevant events from an Event Knowledge Graph. "
    "Your task: perform step-by-step reasoning using only the provided events and return EXACTLY ONE JSON object. "
    "No explanations, no markdown, no extra text."
)

AGENT_USER_TMPL = """Question: {question}

Relevant Events (JSON):
{events_json}

Return EXACTLY ONE JSON object with:
- reasoning_steps: list of objects
  - step: integer (starting from 1, incrementing by 1)
  - action: string (short description of the reasoning step)
  - used_events: list of event IDs from the provided events
  - notes: string (brief explanation of this step)
- answer: string (final concise answer to the question, based ONLY on the provided events)

Strict rules:
- Output ONLY the JSON object, starting with '{{' and ending with '}}'.
- Do NOT output markdown, comments, or multiple JSON objects.
- All 'used_events' IDs MUST match IDs from the provided events_json.
- Keep all strings concise and relevant.
- If you cannot answer, set 'answer' to "Insufficient information" and leave reasoning_steps empty.
"""

def extract_from_chunk(llm_invoker: Any, chunk: str) -> Tuple[List[Event], List[Entity], List[Relation]]:
    print(f"Extracting from chunk (length={len(chunk)})...")
    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM},
        {"role": "user", "content": EXTRACTION_USER_TMPL.format(chunk=chunk)},
    ]
    raw = llm_invoker(messages=messages)

    # --- robust JSON parse for string outputs ---
    preview = str(raw)
    print("LLM raw output (preview, first 200 chars):", preview)

    def ensure_json(obj: Any) -> Dict[str, Any]:
        if isinstance(obj, dict):
            return obj
        s = str(obj).strip()
        if not s:
            print("LLM returned empty string. Cannot parse JSON.")
            raise ValueError("Empty LLM output")
        try:
            return json.loads(s)
        except Exception:
            import re
            m = re.search(r"\{[\s\S]*\}", s)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    print("Failed to parse extracted JSON block. First 300 chars:\n", m.group(0)[:300])
                    raise
            print("No JSON found in LLM output. First 300 chars:\n", s[:300])
            raise

    payload = ensure_json(raw)

    ents_in = payload.get("entities", []) or []
    evs_in = payload.get("events", []) or []
    rels_in = payload.get("relations", []) or []
    print(f"Parsed JSON -> entities={len(ents_in)}, events={len(evs_in)}, relations={len(rels_in)}")

    events, entities, relations = [], [], []
    name_to_id: Dict[str, str] = {}

    for e in ents_in:
        ent = Entity(name=e.get("name", "").strip(), type=e.get("type", "Other"), aliases=e.get("aliases", []))
        entities.append(ent)
        name_to_id[ent.name] = ent.id
        for alias in ent.aliases:
            name_to_id[alias] = ent.id

    for ev in evs_in:
        parts = [name_to_id.get(p) for p in ev.get("participants", []) if name_to_id.get(p)]
        events.append(Event(title=ev.get("title", "").strip(),
                            description=ev.get("description", "").strip(),
                            time=ev.get("time"),
                            participants=parts))

    for r in rels_in:
        relations.append(Relation(source=r.get("from_title", "").strip(),
                                  target=r.get("to_title", "").strip(),
                                  type=r.get("type", "related_to"),
                                  evidence="from_chunk"))
    return events, entities, relations

def build_ekg(text_chunker: Any, llm_invoker: Any, docs: List[str]) -> EventKnowledgeGraph:
    print("Building EKG from documents...")
    ekg = EventKnowledgeGraph()
    chunks = []
    for idx, d in enumerate(docs, start=1):
        print(f"Chunking document {idx} (length={len(d)})...")
        doc_chunks = text_chunker.chunk(d, min_length=10)
        print(f" -> got {len(doc_chunks)} chunks")
        chunks.extend([c['text'] for c in doc_chunks])
    tmp_events, tmp_entities, tmp_relations = [], [], []
    for i, ch in enumerate(chunks, start=1):
        print(f"Processing chunk {i}/{len(chunks)})")
        evs, ents, rels = extract_from_chunk(llm_invoker, ch)
        tmp_events.extend(evs)
        tmp_entities.extend(ents)
        tmp_relations.extend(rels)
    print(f"Merging {len(tmp_entities)} entities...")
    name_to_entity_id: Dict[str, str] = {}
    for ent in tmp_entities:
        key = ent.name.casefold()
        if key in name_to_entity_id:
            base_id = name_to_entity_id[key]
            ekg.entities[base_id].aliases.extend(ent.aliases)
        else:
            ekg.add_entity(ent)
            name_to_entity_id[key] = ent.id
    print(f"Adding {len(tmp_events)} events...")
    for ev in tmp_events:
        ekg.add_event(ev)
    print("Mapping relations...")
    title_to_id = {e.title: e.id for e in ekg.events.values()}
    fixed_relations = []
    for r in tmp_relations:
        s, t = title_to_id.get(r.source), title_to_id.get(r.target)
        if s and t and s != t:
            r.source, r.target = s, t
            fixed_relations.append(r)
    ekg.relations = fixed_relations
    print("Embedding events...")
    texts = [f"{e.title}. {e.description}" for e in ekg.events.values()]
    vecs = text_chunker.embedding_func(sentences=texts)
    for ev, v in zip(ekg.events.values(), vecs):
        ev.embedding = list(map(float, v))
    return ekg

def retrieve_initial_events(embedding_func: Any, ekg: EventKnowledgeGraph, question: str, k: int = 5) -> List[Event]:
    print(f"Retrieving top-{k} events for the question...")
    q_vecs = embedding_func(sentences=[question])
    q_vec = list(map(float, q_vecs[0]))
    scored = [(cosine_sim(q_vec, ev.embedding), ev) for ev in ekg.events.values()]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [ev for _, ev in scored[:k]]

def expand_with_neighbors(seed_events: List[Event], ekg: EventKnowledgeGraph, max_neighbors: int = 10) -> List[Event]:
    print("Expanding context with neighbors...")
    res = {e.id: e for e in seed_events}
    for e in list(seed_events):
        for n in ekg.neighbors(e.id)[:max_neighbors]:
            res[n.id] = n
    return list(res.values())

def agent_reasoning(llm_invoker: Any, question: str, events: List[Event]) -> Dict[str, Any]:
    print("Running agent reasoning...")
    ev_payload = [{"id": e.id, "title": e.title, "description": e.description, "time": e.time, "participants": e.participants} for e in events]
    messages = [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": AGENT_USER_TMPL.format(question=question, events_json=json.dumps(ev_payload, ensure_ascii=False))},
    ]
    raw = llm_invoker(messages=messages, max_new_tokens=6000)
    print('#'*10)
    print(raw)
    return raw if isinstance(raw, dict) else json.loads(raw)

def run_pipeline(docs: List[str], question: str, text_chunker: Any, llm_invoker: Any, topk: int = 5) -> Dict[str, Any]:
    print("Starting EventRAG pipeline...")
    ekg = build_ekg(text_chunker, llm_invoker, docs)
    seeds = retrieve_initial_events(text_chunker.embedding_func, ekg, question, k=topk)
    print(f"Initial events retrieved: {len(seeds)}")
    context_events = expand_with_neighbors(seeds, ekg)
    print(f"Context events total: {len(context_events)}")
    agent_out = agent_reasoning(llm_invoker, question, context_events)
    print("Pipeline finished.")
    return {"question": question, "events_used": [e.id for e in context_events], "agent_output": agent_out}

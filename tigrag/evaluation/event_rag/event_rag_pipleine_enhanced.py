"""
EventRAG – Enhanced pipeline (provenance-preserving) implementing:
- semantic clustering (entities & events) with θ thresholds
- provenance-preserving merge (no hard deletions), persistent `similar` edges
- dual-stream vector indices: RAW (all items) + CONSOLIDATED (cluster reps)
- knowledge expansion (LLM-assisted) for underconnected nodes
- temporal parsing with simple interval support + precedes/overlaps edges
- iterative agent loop with reflection/consistency checks
- lightweight sub-query planner for agentic retrieval
- pluggable vector index (in-memory) separate from consolidated EKG
- verbose console logging at key LLM and pipeline touchpoints

This file is designed as a drop-in extension of the user's original code.
You can import the existing Entity/Event/Relation/EventKnowledgeGraph types
or use the compatible dataclasses defined here when running standalone.

Public entry point: run_pipeline_enhanced(...)

Assumptions about external deps:
- text_chunker provides: .chunk(text, min_length=...), .embedding_func(sentences=[...])
- llm_invoker(messages=[...], **kwargs) follows OpenAI/LLM chat API style

Author's note: all features are implemented without third-party stores, but
the VectorIndex API allows swapping in a Milvus/FAISS adapter later.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Iterable, Callable, DefaultDict
import uuid
import json
import math
import re
from collections import defaultdict
from datetime import datetime

# ---------------------------
# Console logging helpers
# ---------------------------

def _truncate(s: str, n: int = 800) -> str:
    s = s if isinstance(s, str) else str(s)
    return s if len(s) <= n else s[: n - 3] + "..."

def log_stage(title: str, payload: Optional[Any] = None):
    print("\n[EventRAG] " + title)
    if payload is not None:
        try:
            print(_truncate(json.dumps(payload, ensure_ascii=False, indent=2)))
        except Exception:
            print(_truncate(str(payload)))

# ---------------------------
# Core dataclasses (compatible + provenance/meta fields)
# ---------------------------

@dataclass
class Entity:
    name: str
    type: str
    aliases: List[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: f"EN_{uuid.uuid4().hex[:8]}")
    # provenance/merge metadata
    canonical_id: Optional[str] = None  # cluster representative id
    cluster_id: Optional[str] = None    # stable cluster label

@dataclass
class Event:
    title: str
    description: str
    time: Optional[str] = None  # raw string input
    participants: List[str] = field(default_factory=list)  # entity IDs
    id: str = field(default_factory=lambda: f"EV_{uuid.uuid4().hex[:8]}")
    embedding: Optional[List[float]] = None
    # derived temporal fields (set by temporal parsing)
    time_parsed: Optional[datetime] = None
    # interval support
    time_start: Optional[datetime] = None
    time_end: Optional[datetime] = None
    # provenance/merge metadata
    canonical_id: Optional[str] = None
    cluster_id: Optional[str] = None

@dataclass
class Relation:
    source: str  # event ID (after mapping)
    target: str  # event ID (after mapping)
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
        # Prevent exact duplicates
        for r in self.relations:
            if r.source == rel.source and r.target == rel.target and r.type == rel.type:
                return
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

# ---------------------------
# Utility: cosine similarity
# ---------------------------

def cosine_sim(a: Optional[List[float]], b: Optional[List[float]]) -> float:
    if not a or not b:
        return -1.0
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a)) or 1e-12
    nb = math.sqrt(sum(y*y for y in b)) or 1e-12
    return dot / (na * nb)

# ---------------------------
# Vector index (pluggable)
# ---------------------------

class VectorIndex:
    """Simple in-memory vector index for (id -> vector). Replace with Milvus/FAISS to scale."""
    def __init__(self, embed_fn: Callable[[List[str]], List[List[float]]]):
        self._vecs: Dict[str, List[float]] = {}
        self._embed_fn = embed_fn

    def add(self, ids: List[str], texts: List[str], vecs: Optional[List[List[float]]] = None):
        if vecs is None:
            vecs = self._embed_fn(texts)
        for i, vid in enumerate(ids):
            self._vecs[vid] = list(map(float, vecs[i]))

    def upsert(self, ids: List[str], texts: List[str]):
        self.add(ids, texts)

    def query(self, query_text: str, topk: int = 5) -> List[Tuple[str, float]]:
        qv = list(map(float, self._embed_fn([query_text])[0]))
        scored = [(cosine_sim(qv, v), vid) for vid, v in self._vecs.items()]
        scored.sort(key=lambda t: t[0], reverse=True)
        return [(vid, score) for score, vid in scored[:topk]]

    def get(self, vid: str) -> Optional[List[float]]:
        return self._vecs.get(vid)

# ---------------------------
# Temporal parsing + edge derivation (with simple intervals)
# ---------------------------

_TIME_PATTERNS = [
    # ISO date
    (re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"), "%Y-%m-%d"),
    # Common textual formats
    (re.compile(r"\b([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})\b"), "%B %d, %Y"),
    (re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b"), "%d.%m.%Y"),
]

_MONTHS_DE = {
    'januar': 'January', 'februar': 'February', 'märz': 'March', 'maerz': 'March', 'april': 'April',
    'mai': 'May', 'juni': 'June', 'juli': 'July', 'august': 'August', 'september': 'September',
    'oktober': 'October', 'november': 'November', 'dezember': 'December'
}

_MONTHS_ALL = {
    **{k: v for k, v in _MONTHS_DE.items()},
    **{m.lower(): m for m in ["January","February","March","April","May","June","July","August","September","October","November","December"]}
}

_MONTH_RE = r"(" + "|".join([re.escape(m) for m in _MONTHS_ALL.keys()]) + r")"
_RANGE_PATTERNS = [
    # "June–July 2024" or "Juni–Juli 2024"
    re.compile(rf"\b{_MONTH_RE}[\-\–\u2013]{_MONTH_RE}\s+(\d{{4}})\b", re.IGNORECASE),
    # "2020–2022"
    re.compile(r"\b(\d{4})[\-\–\u2013](\d{4})\b"),
]


def _normalize_date_text(s: str) -> str:
    low = s.strip().lower()
    for de, en in _MONTHS_DE.items():
        low = re.sub(rf"\b{re.escape(de)}\b", en.lower(), low)
    return low


def parse_time_to_interval(raw: Optional[str]) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Return (start,end) if a range is detected, or (point, None) for single date/year."""
    if not raw:
        return None, None
    s = raw.strip()
    if not s:
        return None, None

    # explicit single-date patterns
    for rx, fmt in _TIME_PATTERNS:
        if rx.search(s):
            try:
                if fmt == "%d.%m.%Y":
                    m = rx.search(s)
                    day, month, year = m.group(1), m.group(2), m.group(3)
                    s2 = f"{day}.{month}.{year}"
                    dt = datetime.strptime(s2, "%d.%m.%Y")
                    return dt, None
                if fmt == "%B %d, %Y":
                    s2 = _normalize_date_text(s)
                    s2 = re.sub(r"^([a-z])", lambda m: m.group(1).upper(), s2)
                    dt = datetime.strptime(s2, "%B %d, %Y")
                    return dt, None
                dt = datetime.strptime(s, fmt)
                return dt, None
            except Exception:
                pass

    # range: Month–Month Year
    m0 = _RANGE_PATTERNS[0].search(s)
    if m0:
        m1, m2, year = m0.group(1).lower(), m0.group(2).lower(), int(m0.group(3))
        m1_en = _MONTHS_ALL[m1]
        m2_en = _MONTHS_ALL[m2]
        start = datetime.strptime(f"{m1_en} 1, {year}", "%B %d, %Y")
        # end = last day of m2: approximate by next month -1 day
        if m2_en == "December":
            end = datetime(year, 12, 31)
        else:
            months = ["January","February","March","April","May","June","July","August","September","October","November","December"]
            idx = months.index(m2_en)
            nxt = datetime(year + (1 if idx == 11 else 0), 1 if idx == 11 else idx + 2, 1)
            end = nxt.replace(day=1) - (datetime.min.replace(year=1, month=1, day=2) - datetime.min.replace(year=1, month=1, day=1))
        return start, end

    # range: YYYY–YYYY
    m1 = _RANGE_PATTERNS[1].search(s)
    if m1:
        y1, y2 = int(m1.group(1)), int(m1.group(2))
        if y2 < y1:
            y1, y2 = y2, y1
        return datetime(y1, 1, 1), datetime(y2, 12, 31)

    # year-only fallback
    m = re.search(r"\b(\d{4})\b", s)
    if m:
        y = int(m.group(1))
        return datetime(y, 1, 1), None

    return None, None


def derive_temporal_edges(ekg: EventKnowledgeGraph, evidence: str = "temporal_infer") -> List[Relation]:
    """Create precedes/overlaps edges based on parsed datetimes/intervals. Avoid duplicates."""
    for ev in ekg.events.values():
        start, end = parse_time_to_interval(ev.time)
        ev.time_start, ev.time_end = start, end
        ev.time_parsed = start or ev.time_parsed

    existing = {(r.source, r.target, r.type) for r in ekg.relations}
    derived: List[Relation] = []

    evs = list(ekg.events.values())
    for i in range(len(evs)):
        for j in range(i+1, len(evs)):
            a, b = evs[i], evs[j]
            as_, ae = a.time_start, a.time_end or a.time_start
            bs_, be = b.time_start, b.time_end or b.time_start
            if as_ and bs_:
                if ae and be and (as_ <= be and bs_ <= ae):
                    # intervals overlap
                    tup = (a.id, b.id, "overlaps")
                    if tup not in existing:
                        derived.append(Relation(source=a.id, target=b.id, type="overlaps", evidence=evidence))
                        existing.add(tup)
                elif ae and bs_ and ae < bs_:
                    tup = (a.id, b.id, "precedes")
                    if tup not in existing:
                        derived.append(Relation(source=a.id, target=b.id, type="precedes", evidence=evidence))
                        existing.add(tup)
                elif be and as_ and be < as_:
                    tup = (b.id, a.id, "precedes")
                    if tup not in existing:
                        derived.append(Relation(source=b.id, target=a.id, type="precedes", evidence=evidence))
                        existing.add(tup)
    return derived

# ---------------------------
# Semantic clustering (entities + events) – provenance-preserving
# ---------------------------

@dataclass
class MergeConfig:
    entity_theta: float = 0.86
    event_theta: float = 0.84
    create_similar_edges: bool = True

class SemanticMerger:
    def __init__(self, embed_fn: Callable[[List[str]], List[List[float]]], cfg: MergeConfig = MergeConfig()):
        self.embed_fn = embed_fn
        self.cfg = cfg

    def _embed(self, texts: List[str]) -> List[List[float]]:
        return [list(map(float, v)) for v in self.embed_fn(texts)]

    def cluster_entities(self, ekg: EventKnowledgeGraph) -> Dict[str, str]:
        """Return mapping entity_id -> cluster_id (canonical chosen as first seen). No deletions."""
        ents = list(ekg.entities.values())
        if not ents:
            return {}
        names = [e.name for e in ents]
        vecs = self._embed(names)
        cluster_id_of: Dict[str, str] = {}
        canonical_of_cluster: Dict[str, str] = {}
        used = set()
        for i, ei in enumerate(ents):
            if ei.id in used:
                continue
            cluster = f"CEN_{uuid.uuid4().hex[:8]}"
            canonical = ei.id
            canonical_of_cluster[cluster] = canonical
            cluster_id_of[ei.id] = cluster
            used.add(ei.id)
            for j in range(i+1, len(ents)):
                ej = ents[j]
                if ej.id in used:
                    continue
                sim = cosine_sim(vecs[i], vecs[j])
                if sim >= self.cfg.entity_theta:
                    cluster_id_of[ej.id] = cluster
                    used.add(ej.id)
        # annotate metadata
        for eid, ent in ekg.entities.items():
            cl = cluster_id_of.get(eid, f"CEN_{uuid.uuid4().hex[:8]}")
            ent.cluster_id = cl
            ent.canonical_id = canonical_of_cluster.get(cl, eid)
        return cluster_id_of

    def cluster_events(self, ekg: EventKnowledgeGraph) -> Dict[str, str]:
        """Return mapping event_id -> cluster_id. Create persistent `similar` edges between cluster members."""
        evs = list(ekg.events.values())
        if not evs:
            return {}
        texts = [f"{e.title}. {e.description}".strip() for e in evs]
        vecs = self._embed(texts)
        for ev, v in zip(evs, vecs):
            ev.embedding = v
        cluster_id_of: Dict[str, str] = {}
        canonical_of_cluster: Dict[str, str] = {}
        used = set()
        for i, ei in enumerate(evs):
            if ei.id in used:
                continue
            cluster = f"CEV_{uuid.uuid4().hex[:8]}"
            canonical = ei.id
            canonical_of_cluster[cluster] = canonical
            cluster_id_of[ei.id] = cluster
            used.add(ei.id)
            for j in range(i+1, len(evs)):
                ej = evs[j]
                if ej.id in used:
                    continue
                sim = cosine_sim(vecs[i], vecs[j])
                if sim >= self.cfg.event_theta:
                    cluster_id_of[ej.id] = cluster
                    used.add(ej.id)
                    if self.cfg.create_similar_edges:
                        ekg.add_relation(Relation(source=canonical, target=ej.id, type="similar", evidence="merge_similar"))
        # annotate metadata
        for eid, ev in ekg.events.items():
            cl = cluster_id_of.get(eid, f"CEV_{uuid.uuid4().hex[:8]}")
            ev.cluster_id = cl
            ev.canonical_id = canonical_of_cluster.get(cl, eid)
        return cluster_id_of

# ---------------------------
# Knowledge expansion (LLM-assisted) – with logging
# ---------------------------

EXPAND_SYSTEM = (
    "You are a knowledge graph completion assistant. Given a set of events and entities, "
    "identify likely missing relations (precedes, follows, causes, related_to, similar) and "
    "potentially missing participants between existing events. Respond with ONE JSON object only."
)

EXPAND_USER_TMPL = """Context Events (JSON):\n{events_json}\n\nEntities (JSON):\n{entities_json}\n\nReturn ONLY a JSON object with keys:\n- relations: list of objects with fields [from_id, to_id, type, evidence]\n- participant_fills: list of objects with fields [event_id, entity_name]\nRules:\n- Use only event IDs provided.\n- If unsure, omit.\n- Keep concise evidence.\n"""


def knowledge_expansion(llm_invoker: Any, ekg: EventKnowledgeGraph, event_scope: Optional[List[str]] = None) -> None:
    events = [ekg.events[eid] for eid in (event_scope or list(ekg.events.keys())) if eid in ekg.events]
    ev_payload = [
        {"id": e.id, "title": e.title, "desc": e.description, "time": e.time, "participants": e.participants}
        for e in events
    ]
    ent_payload = [
        {"id": en.id, "name": en.name, "type": en.type, "aliases": en.aliases}
        for en in ekg.entities.values()
    ]
    messages = [
        {"role": "system", "content": EXPAND_SYSTEM},
        {"role": "user", "content": EXPAND_USER_TMPL.format(
            events_json=json.dumps(ev_payload, ensure_ascii=False),
            entities_json=json.dumps(ent_payload, ensure_ascii=False)
        )},
    ]
    log_stage("LLM invoke: knowledge_expansion::messages", messages)
    raw = llm_invoker(messages=messages)
    log_stage("LLM output: knowledge_expansion::raw", raw)

    try:
        payload = raw if isinstance(raw, dict) else json.loads(str(raw))
    except Exception:
        m = re.search(r"\{[\s\S]*\}", str(raw))
        if not m:
            return
        payload = json.loads(m.group(0))

    for r in payload.get("relations", []) or []:
        fr, to, t = r.get("from_id"), r.get("to_id"), r.get("type", "related_to")
        if fr in ekg.events and to in ekg.events and fr != to:
            ekg.add_relation(Relation(source=fr, target=to, type=t, evidence=r.get("evidence", "expand")))
    name_to_id = {en.name: en.id for en in ekg.entities.values()}
    for p in payload.get("participant_fills", []) or []:
        ev_id, en_name = p.get("event_id"), p.get("entity_name")
        if ev_id in ekg.events and en_name in name_to_id:
            pid = name_to_id[en_name]
            if pid not in ekg.events[ev_id].participants:
                ekg.events[ev_id].participants.append(pid)

# ---------------------------
# Agent with iterative loop + reflection + sub-queries
# ---------------------------

AGENT_SYSTEM = (
    "You are an EventRAG reasoning agent. Use ONLY provided events."
    " Return ONE JSON object with reasoning_steps and answer."
)

AGENT_USER_TMPL = """Question: {question}\n\nRelevant Events (JSON):\n{events_json}\n\nReturn EXACTLY ONE JSON object with:\n- reasoning_steps: list of objects\n  - step: integer\n  - action: string\n  - used_events: list of event IDs\n  - notes: string\n- answer: string\nRules:\n- If insufficient evidence, set answer to \"Insufficient information\".\n- Be concise.\n"""

REFLECT_SYSTEM = (
    "You are a strict evaluator. Given a question, events and a candidate answer,"
    " check for temporal/logical inconsistencies or missing support."
    " Reply with ONE JSON object: {valid: bool, reasons: string}."
)

REFLECT_USER_TMPL = """Question: {question}\nAnswer: {answer}\n\nEvents (JSON):\n{events_json}\n\nReturn ONLY one JSON object with keys [valid, reasons]."""

SUBQ_SYSTEM = (
    "You split a question into 2-4 concise sub-queries for Event-KG retrieval."
    " Focus on participants, temporal aspects, causes/effects, and key entities."
    " Return ONLY a JSON list of strings."
)

SUBQ_USER_TMPL = "Split this question into 2-4 short retrieval sub-queries (JSON list only):\n{q}"


def make_subqueries(llm_invoker: Any, question: str) -> List[str]:
    messages = [
        {"role": "system", "content": SUBQ_SYSTEM},
        {"role": "user", "content": SUBQ_USER_TMPL.format(q=question)},
    ]
    log_stage("LLM invoke: subquery_planner::messages", messages)
    raw = llm_invoker(messages=messages)
    log_stage("LLM output: subquery_planner::raw", raw)
    try:
        arr = raw if isinstance(raw, list) else json.loads(str(raw))
        if isinstance(arr, list) and all(isinstance(x, str) for x in arr):
            return arr[:4]
    except Exception:
        pass
    # safe fallback
    return [question, f"Temporal aspects of: {question}", f"Participants/entities in: {question}"]


def agent_reasoning_once(llm_invoker: Any, question: str, events: List[Event], max_new_tokens: int = 2000) -> Dict[str, Any]:
    ev_payload = [{"id": e.id, "title": e.title, "description": e.description, "time": e.time, "participants": e.participants} for e in events]
    messages = [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": AGENT_USER_TMPL.format(question=question, events_json=json.dumps(ev_payload, ensure_ascii=False))},
    ]
    log_stage("LLM invoke: agent_reasoning::messages", messages)
    raw = llm_invoker(messages=messages, max_new_tokens=max_new_tokens)
    log_stage("LLM output: agent_reasoning::raw", raw)
    return raw if isinstance(raw, dict) else json.loads(str(raw))


def reflect_answer(llm_invoker: Any, question: str, events: List[Event], answer: str) -> Tuple[bool, str]:
    ev_payload = [{"id": e.id, "title": e.title, "time": e.time} for e in events]
    messages = [
        {"role": "system", "content": REFLECT_SYSTEM},
        {"role": "user", "content": REFLECT_USER_TMPL.format(question=question, answer=answer, events_json=json.dumps(ev_payload, ensure_ascii=False))},
    ]
    log_stage("LLM invoke: reflection::messages", messages)
    raw = llm_invoker(messages=messages)
    log_stage("LLM output: reflection::raw", raw)
    try:
        obj = raw if isinstance(raw, dict) else json.loads(str(raw))
        valid = bool(obj.get("valid", False))
        reasons = str(obj.get("reasons", ""))
        return valid, reasons
    except Exception:
        return False, "Reflection parse error"

# ---------------------------
# Enhanced build + retrieval + loop (provenance-preserving)
# ---------------------------

EXTRACT_SYSTEM = (
    "You extract structured event data for an Event Knowledge Graph. "
    "Follow the schema EXACTLY. Output ONE JSON object only. No explanations."
)

EXTRACT_USER_TMPL = """Read the text and return EXACTLY ONE JSON object with keys: events, entities, relations.\n\nSchema:\n- events: list of objects\n  - title: string\n  - description: string\n  - time: string or null\n  - participants: list of strings (entity names from 'entities')\n- entities: list of objects\n  - name: string\n  - type: one of [\"Person\",\"Organisation\",\"Location\",\"Other\"]\n  - aliases: list of strings (optional)\n- relations: list of objects\n  - from_title: string (must match an 'events[].title')\n  - to_title: string (must match an 'events[].title')\n  - type: one of [\"precedes\",\"follows\",\"causes\",\"related_to\",\"similar\"]\n\nStrict rules:\n- Output ONE and ONLY ONE top-level JSON object.\n- 'participants' MUST be names present in 'entities'.\n- If uncertain, omit the field or use null.\n- Keep values concise.\n\nText:\n<<<\n{chunk}\n>>>\n\nReturn ONLY the JSON object. Start with '{{' and end with '}}'.\n"""


def extract_from_chunk(llm_invoker: Any, chunk: str) -> Tuple[List[Event], List[Entity], List[Relation]]:
    messages = [
        {"role": "system", "content": EXTRACT_SYSTEM},
        {"role": "user", "content": EXTRACT_USER_TMPL.format(chunk=chunk)},
    ]
    log_stage("LLM invoke: extract_from_chunk::messages", messages)
    raw = llm_invoker(messages=messages)
    log_stage("LLM output: extract_from_chunk::raw", raw)

    def ensure_json(obj: Any) -> Dict[str, Any]:
        if isinstance(obj, dict):
            return obj
        s = str(obj).strip()
        if not s:
            raise ValueError("Empty LLM output")
        try:
            return json.loads(s)
        except Exception:
            m = re.search(r"\{[\s\S]*\}", s)
            if m:
                return json.loads(m.group(0))
            raise

    payload = ensure_json(raw)
    ents_in = payload.get("entities", []) or []
    evs_in = payload.get("events", []) or []
    rels_in = payload.get("relations", []) or []

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

# Build EKG with clustering/temporal/expand

def build_ekg_enhanced(text_chunker: Any, llm_invoker: Any, docs: List[str], merge_cfg: MergeConfig = MergeConfig()) -> EventKnowledgeGraph:
    ekg = EventKnowledgeGraph()
    chunks: List[str] = []
    # 1) Chunk docs
    for d in docs:
        for c in text_chunker.chunk(d, min_length=10):
            chunks.append(c['text'])
    log_stage("Chunking complete", {"num_chunks": len(chunks)})

    tmp_events: List[Event] = []
    tmp_entities: List[Entity] = []
    tmp_relations: List[Relation] = []

    # 2) Extract
    for ch in chunks:
        evs, ents, rels = extract_from_chunk(llm_invoker, ch)
        tmp_events.extend(evs)
        tmp_entities.extend(ents)
        tmp_relations.extend(rels)

    # 3) Add entities (raw)
    for ent in tmp_entities:
        ekg.add_entity(ent)

    # 4) Add events (raw)
    for ev in tmp_events:
        ekg.add_event(ev)

    # 5) Map relations by event titles to IDs (no deletions)
    title_to_id = {e.title: e.id for e in ekg.events.values()}
    fixed_relations: List[Relation] = []
    for r in tmp_relations:
        s, t = title_to_id.get(r.source), title_to_id.get(r.target)
        if s and t and s != t:
            fixed_relations.append(Relation(source=s, target=t, type=r.type, evidence=r.evidence))
    ekg.relations.extend(fixed_relations)

    # 6) Semantic clustering (no hard merge)
    merger = SemanticMerger(text_chunker.embedding_func, merge_cfg)
    ent_clusters = merger.cluster_entities(ekg)
    ev_clusters = merger.cluster_events(ekg)
    log_stage("Semantic clustering complete", {"entity_clusters": len(set(ent_clusters.values())), "event_clusters": len(set(ev_clusters.values()))})

    # 7) Embed any events lacking vectors
    need_ids, need_texts = [], []
    for ev in ekg.events.values():
        if not ev.embedding:
            need_ids.append(ev.id)
            need_texts.append(f"{ev.title}. {ev.description}")
    if need_ids:
        vecs = text_chunker.embedding_func(sentences=need_texts)
        for i, ev_id in enumerate(need_ids):
            ekg.events[ev_id].embedding = list(map(float, vecs[i]))
    log_stage("Event embeddings updated", {"added": len(need_ids)})

    # 8) Temporal parsing + derived edges
    derived = derive_temporal_edges(ekg)
    ekg.relations.extend(derived)
    log_stage("Temporal edges derived", {"added": len(derived)})

    # 9) Knowledge expansion (optional – scoped to all consolidated events)
    knowledge_expansion(llm_invoker, ekg, event_scope=list(ekg.events.keys()))

    return ekg

# Retrieval helpers

class ConsolidatedIndex:
    """Dual-stream indices: raw and consolidated (cluster reps)."""
    def __init__(self, embed_fn: Callable[[List[str]], List[List[float]]] ):
        self.raw = VectorIndex(embed_fn)
        self.cons = VectorIndex(embed_fn)
        # map consolidated-id -> member event ids
        self.cluster_members: Dict[str, List[str]] = {}

    def build_from_ekg(self, ekg: EventKnowledgeGraph):
        # RAW: index every event
        raw_ids = list(ekg.events.keys())
        raw_texts = [f"{ekg.events[i].title}. {ekg.events[i].description}" for i in raw_ids]
        self.raw.upsert(raw_ids, raw_texts)

        # CONS: one rep per cluster, aggregate member titles/descriptions
        by_cluster: DefaultDict[str, List[Event]] = defaultdict(list)
        for ev in ekg.events.values():
            by_cluster[ev.cluster_id or ev.id].append(ev)
        cons_ids, cons_texts = [], []
        for cl, members in by_cluster.items():
            # Representative is the canonical event
            canon = None
            for m in members:
                if m.id == (m.canonical_id or m.id):
                    canon = m
                    break
            canon = canon or members[0]
            cons_id = f"CONS_{cl}"
            self.cluster_members[cons_id] = [m.id for m in members]
            # aggregate text for better recall
            agg = ". ".join([f"{m.title}. {m.description}".strip() for m in members])
            cons_ids.append(cons_id)
            cons_texts.append(agg)
        self.cons.upsert(cons_ids, cons_texts)

    def query_consolidated(self, q: str, topk: int = 5) -> List[str]:
        hits = self.cons.query(q, topk=topk)
        ids: List[str] = []
        for cons_id, _ in hits:
            ids.extend(self.cluster_members.get(cons_id, []))
        # unique preserve order
        seen = set()
        out = []
        for i in ids:
            if i not in seen:
                seen.add(i)
                out.append(i)
        return out


def retrieve_events(cons_index: ConsolidatedIndex, ekg: EventKnowledgeGraph, queries: List[str], k: int = 5) -> List[Event]:
    hits: List[str] = []
    for q in queries:
        hits.extend(cons_index.query_consolidated(q, topk=k))
    # unique preserve order
    seen = set()
    uniq = []
    for vid in hits:
        if vid in ekg.events and vid not in seen:
            uniq.append(vid)
            seen.add(vid)
    return [ekg.events[vid] for vid in uniq]


def expand_with_neighbors(seed_events: List[Event], ekg: EventKnowledgeGraph, max_neighbors: int = 8) -> List[Event]:
    res: Dict[str, Event] = {e.id: e for e in seed_events}
    for e in list(seed_events):
        for n in ekg.neighbors(e.id)[:max_neighbors]:
            res[n.id] = n
    return list(res.values())

# ---------------------------
# Main iterative loop
# ---------------------------

@dataclass
class LoopConfig:
    topk: int = 5
    max_neighbors: int = 8
    max_iters: int = 3
    reflect: bool = True


def run_pipeline_enhanced(
    docs: List[str],
    question: str,
    text_chunker: Any,
    llm_invoker: Any,
    merge_cfg: MergeConfig = MergeConfig(),
    loop_cfg: LoopConfig = LoopConfig(),
) -> Dict[str, Any]:
    """Builds enhanced EKG, constructs dual vector indices, runs iterative agent loop with reflection."""
    log_stage("Pipeline start", {"question": question, "num_docs": len(docs)})

    # Build consolidated/expanded graph
    ekg = build_ekg_enhanced(text_chunker, llm_invoker, docs, merge_cfg)
    log_stage("EKG built", {"events": len(ekg.events), "entities": len(ekg.entities), "relations": len(ekg.relations)})

    # Build dual indices
    cindex = ConsolidatedIndex(text_chunker.embedding_func)
    cindex.build_from_ekg(ekg)
    log_stage("Vector indices built", {"raw_items": len(ekg.events), "clusters": len(cindex.cluster_members)})

    # Plan sub-queries
    subqs = make_subqueries(llm_invoker, question)
    log_stage("Sub-queries", subqs)

    used_event_ids: List[str] = []
    final_answer = "Insufficient information"
    trace: List[Dict[str, Any]] = []
    context_events: List[Event] = []

    for iteration in range(1, loop_cfg.max_iters + 1):
        seeds = retrieve_events(cindex, ekg, subqs, k=loop_cfg.topk)
        context_events = expand_with_neighbors(seeds, ekg, max_neighbors=loop_cfg.max_neighbors)
        used_event_ids = [e.id for e in context_events]
        log_stage(f"Iteration {iteration}: context size", {"seeds": len(seeds), "context": len(context_events)})

        agent_out = agent_reasoning_once(llm_invoker, question, context_events)
        final_answer = str(agent_out.get("answer", "Insufficient information"))
        trace.append({"iter": iteration, "agent": agent_out})

        if loop_cfg.reflect:
            valid, reasons = reflect_answer(llm_invoker, question, context_events, final_answer)
            trace[-1]["reflection"] = {"valid": valid, "reasons": reasons}
            log_stage(f"Iteration {iteration}: reflection", trace[-1]["reflection"])
            if valid:
                break
            # If invalid, try to expand knowledge and iterate again
            knowledge_expansion(llm_invoker, ekg, event_scope=used_event_ids)
            # re-derive temporal edges (in case new times added)
            added = derive_temporal_edges(ekg, evidence="temporal_recheck")
            ekg.relations.extend(added)
            log_stage(f"Iteration {iteration}: post-reflection expansion", {"new_temporal_edges": len(added)})
            # refresh indices
            cindex.build_from_ekg(ekg)
        else:
            break

    result = {
        "question": question,
        "events_used": used_event_ids,
        "answer": final_answer,
        "iterations": len(trace),
        "trace": trace,
    }
    log_stage("Pipeline end::result", result)
    return result

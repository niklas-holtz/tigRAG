from event_rag.event_rag_pipeline import run_pipeline
from event_rag.event_rag_pipleine_enhanced import run_pipeline_enhanced, MergeConfig, LoopConfig
from tigrag.utils.embedding_invoker import EmbeddingInvoker
from tigrag.utils.llm_invoker import LLMInvoker
from tigrag.nlp.text_chunker import TextChunker

embedder = EmbeddingInvoker()
text_chunker = TextChunker(embedder)
llm = LLMInvoker()

docs = [open("doc1.txt","r",encoding="utf-8").read(),
        open("doc2.txt","r",encoding="utf-8").read()]

result = run_pipeline_enhanced(
    docs=docs,
    question="Wer sind die Hauptakteure und wie hängen die Ereignisse zeitlich zusammen?",
    text_chunker=text_chunker,
    llm_invoker=llm,
    merge_cfg=MergeConfig(entity_theta=0.85, event_theta=0.83),
    loop_cfg=LoopConfig(topk=6, max_neighbors=10, max_iters=4, reflect=True)
)
print(result)
"""
result = run_pipeline(
    docs=docs,
    question="Wer sind die Hauptakteure und wie hängen die Ereignisse zeitlich zusammen?",
    text_chunker=text_chunker,
    llm_invoker=llm,
    topk=5,
)
print(result)
"""

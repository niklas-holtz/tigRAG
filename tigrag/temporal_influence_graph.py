from .utils.embedding_invoker import EmbeddingInvoker
from .utils.tig_param import TigParam
from .nlp.text_chunker import TextChunker
import os

class TemporalInfluenceGraph:
    def __init__(self, working_dir: str, query_param: TigParam):
        self.working_dir = working_dir

        # Create the embedding function based on the model name
        self.working_dir = working_dir

        os.makedirs(self.working_dir, exist_ok=True)

        # Now embedding cache will be stored in working_dir
        self.embedding_func = EmbeddingInvoker(
            model_name=query_param.embedding_model_name,
            cache_dir=self.working_dir
        )

        # Use LLM function from query param
        #self.llm_model_func = query_param.llm_model_func

    def insert(self, text: str):
        text_chunker = TextChunker(self.embedding_func)
        # chunk
        chunks = text_chunker.chunk(text, min_length=10)
        print(chunks)
        # build tfidf combination
        pass

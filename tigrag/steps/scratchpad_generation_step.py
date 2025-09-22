from __future__ import annotations
import logging
import os
from string import Template
from typing import Any, Dict, List
from ..steps.step import Step, RetrieveContext


class ScratchpadGenerationStep(Step):
    """
    Very similar to AnswerGenerationStep, but loads its prompt template from
    'scratchpad_generation_prompt.txt' located in the same directory as this file.
    """

    def __init__(self, top_n: int = 30, prompt_path: str = "../prompts/scratchpad_generation_prompt.txt"):
        self.top_n = int(top_n)
        # Load the prompt template once; will be filled with {query, retrieval_context}
        self.prompt_text: str = self._load_prompt(prompt_path)
        self._prompt_tmpl: Template = Template(self.prompt_text)

    def run(self, ctx: RetrieveContext) -> RetrieveContext:
        # Du kannst hier denselben Code wie in AnswerGenerationStep nehmen:
        # Events laden, sortieren, Kontext bauen usw.
        # Für Kürze nur der Teil, der prompt ausfüllt und LLM aufruft:

        retrieval_context = getattr(ctx, "retrieval_context", "")
        user_query = getattr(ctx, "query", "") or ""

        try:
            filled_prompt = self._prompt_tmpl.safe_substitute(
                query=user_query,
                retrieval_context=retrieval_context
            )
        except Exception as e:
            logging.error(f"Failed to build scratchpad prompt: {e}")
            ctx.prediction_answer = "Failed to build scratchpad prompt."
            return ctx

        try:
            params = {"max_new_tokens": 2000}
            llm_response = ctx.llm_invoker(
                message=[{"role": "user", "content": filled_prompt}],
                parameters=params
            )
            ctx.prediction_answer = llm_response
        except Exception as e:
            logging.error(f"LLM invocation failed for ScratchpadGenerationStep: {e}")
            ctx.prediction_answer = "Scratchpad generation failed due to LLM invocation error."

        return ctx

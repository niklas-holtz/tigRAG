import os
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Callable
import boto3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class LLMInvoker:
    def __init__(
        self,
        model_name: str = "claude4_bedrock",
        working_dir: str = "./runs",
        log_filename_prefix: str = "llm_calls"
    ):
        self.model_name = model_name
        self.working_dir = os.path.abspath(working_dir)
        self.log_dir = os.path.join(self.working_dir, "llm_logs")
        os.makedirs(self.log_dir, exist_ok=True)

        ts = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file = os.path.join(self.log_dir, f"{log_filename_prefix}_{ts}.log")
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(f"# LLM Call Log - Created {datetime.utcnow().isoformat()}\n\n")

        self._clients: Dict[str, Any] = {}

        # Registry: name -> (init_fn, call_fn)
        self.models: Dict[str, Tuple[Callable[..., None], Callable[..., str]]] = {
            "claude4_bedrock": (self.init_bedrock_noop, lambda **kw: self.call_bedrock_model(model_id="eu.anthropic.claude-sonnet-4-20250514-v1:0", **kw)),
            "claude3_7_bedrock": (self.init_bedrock_noop, lambda **kw: self.call_bedrock_model(model_id="eu.anthropic.claude-3-7-sonnet-20250219-v1:0", **kw)),
            "claude3_5_bedrock": (self.init_bedrock_noop, lambda **kw: self.call_bedrock_model(model_id="eu.anthropic.claude-3-5-sonnet-20240620-v1:0", **kw)),
            "llama3.2_3b_local": (self.init_llama3_2_3b_local, self.call_llama3_2_3b_local),
        }

        if model_name not in self.models:
            raise KeyError(f"Unknown model '{model_name}'. Available: {list(self.models.keys())}")

    # ---------------------------
    # Public API
    # ---------------------------
    def init(self, **init_kwargs) -> None:
        init_fn, _ = self.models[self.model_name]
        init_fn(**init_kwargs)

    def __call__(self, *args, **kwargs) -> str:
        _, call_fn = self.models[self.model_name]
        answer = call_fn(*args, **kwargs)
        self._log_call(kwargs.get("message"), answer)
        return answer

    # ---------------------------
    # Bedrock generic
    # ---------------------------
    def init_bedrock_noop(self, **_: Any) -> None:
        """No initialization needed for Bedrock models."""
        return

    def call_bedrock_model(
        self,
        *,
        model_id: str,
        message: List[Dict[str, str]],
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generic Bedrock Claude API call.
        model_id: full Bedrock modelId string
        message: list of dicts with 'role' and 'content'
        parameters: generation parameters dict
        """
        if not message:
            raise Exception('Missing argument "message".')

        if parameters is None:
            parameters = {
                "max_new_tokens": 6028,
                "top_p": 0.95,
                "temperature": 0.8
            }

        system_msg = message[0]['content'] if message and message[0]['role'] == 'system' else ''
        message_body = message[1:] if system_msg else message

        client = boto3.client(service_name='bedrock-runtime', region_name="eu-west-1")
        body = json.dumps({
            "max_tokens": parameters.get("max_new_tokens", 1028),
            "temperature": parameters.get("temperature", 0.8),
            "top_p": parameters.get("top_p", 0.95),
            "system": system_msg,
            "messages": message_body,
            "anthropic_version": "bedrock-2023-05-31"
        })

        try:
            response = client.invoke_model(
                body=body,
                modelId=model_id,
                accept='application/json',
                contentType='application/json'
            )
            model_response = json.loads(response['body'].read())
            return model_response["content"][0]["text"]
        except Exception as e:
            logging.error(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
            raise e

    # ---------------------------
    # Llama-3.2-3B local
    # ---------------------------
    def init_llama3_2_3b_local(
        self,
        model_path: str = "meta-llama/Llama-3.2-3B-Instruct",
        load_in_4bit: bool = True,
        torch_dtype: str = "float16",
        device_map: str = "auto",
        offload_subdir: str = "offload",
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if "llama3.2_3b_local" in self._clients:
            return

        tokenizer_kwargs = tokenizer_kwargs or {}
        model_kwargs = model_kwargs or {}

        dtype = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype
        offload_folder = os.path.join(self.working_dir, offload_subdir)
        os.makedirs(offload_folder, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)

        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=dtype,
            quantization_config=bnb_config,
            offload_folder=offload_folder,
            **model_kwargs
        )

        self._clients["llama3.2_3b_local"] = {"tokenizer": tokenizer, "model": model}

    def call_llama3_2_3b_local(
        self,
        message: List[Dict[str, str]],
        parameters: Optional[Dict[str, Any]] = None,
        **gen_kwargs,
    ) -> str:
        """
        Call local Llama-3.2-3B-Instruct model.

        Args:
            message: chat messages in the OpenAI-style format
            parameters: dict of generation parameters, e.g.
                {
                    "max_new_tokens": 1500,
                    "temperature": 0.4,
                    "top_p": 0.9,
                    "do_sample": True
                }
            **gen_kwargs: additional generation kwargs
        """
        if "llama3.2_3b_local" not in self._clients:
            raise RuntimeError("Model is not initialized. Call invoker.init(...) first.")

        if parameters is None:
            parameters = {
                "max_new_tokens": 1500,
                "temperature": 0.4,
                "top_p": 0.9,
                "do_sample": True
            }

        tokenizer = self._clients["llama3.2_3b_local"]["tokenizer"]
        model = self._clients["llama3.2_3b_local"]["model"]

        prompt = tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                **parameters,
                **gen_kwargs
            )

        new_tokens = out[0][inputs["input_ids"].shape[-1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


    # ---------------------------
    # Logging
    # ---------------------------
    def _log_call(self, message: Any, answer: str) -> None:
        ts = datetime.utcnow().isoformat()
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"## Call at {ts}\n")
            f.write("Prompt:\n")
            try:
                f.write(json.dumps(message, ensure_ascii=False, indent=2))
            except Exception:
                f.write(str(message))
            f.write("\nAnswer:\n")
            f.write((answer or "") + "\n\n")
            f.write("-" * 80 + "\n\n")

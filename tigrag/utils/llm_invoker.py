import os
import logging
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class LLMInvoker:
    def __init__(
        self,
        working_dir: str,
        model_path: str = r"meta-llama/Llama-3.2-3B-Instruct",
        load_in_4bit: bool = True
    ):
        self.working_dir = os.path.abspath(working_dir)
        self.log_dir = os.path.join(self.working_dir, "llm_logs")
        os.makedirs(self.log_dir, exist_ok=True)

        # Log-Datei (einmalig erstellt, danach immer append)
        timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file = os.path.join(self.log_dir, f"llm_calls_{timestamp}.log")
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(f"# LLM Call Log - Created {datetime.utcnow().isoformat()}\n\n")

        logging.info(f"loading model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        bnb_config = None
        if load_in_4bit:
            logging.info("Enabling 4-bit quantization ...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype="float16",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",  # automatisch CPU/GPU aufteilen
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            offload_folder="offload",  # Ordner für CPU/SSD-Teil
        )

    def __call__(
        self,
        messages: list,
        max_new_tokens: int = 2000,
        temperature: float = 0.4,
        top_p: float = 0.9
    ) -> str:
        """
        messages: list of dicts in the format:
        [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
            ...
        ]
        """

        # create chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # tokenize and send to gpu
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # model generates only the new tokens
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )

        # decode only new tokens (exclude prompt)
        new_tokens = generated_ids[0][inputs["input_ids"].shape[-1]:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        answer = answer.strip()

        # Loggen des Calls
        self._log_call(messages, answer)

        return answer

    def _log_call(self, messages: list, answer: str) -> None:
        """Hängt den Prompt und die Antwort an die Logdatei an."""
        timestamp = datetime.utcnow().isoformat()
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"## Call at {timestamp}\n")
            f.write("Prompt:\n")
            for m in messages:
                f.write(f"[{m['role'].upper()}] {m['content']}\n")
            f.write("\nAnswer:\n")
            f.write(answer + "\n\n")
            f.write("-" * 80 + "\n\n")
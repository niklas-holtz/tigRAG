import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import time

class LLMInvoker:
    def __init__(self, model_path: str = r"meta-llama/Llama-3.2-3B-Instruct", load_in_4bit: bool = True):
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

    def __call__(self, messages: list, max_new_tokens: int = 2000, temperature: float = 0.4, top_p: float = 0.9) -> str:
        """
        messages: list of dicts in the format:
        [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
            and so on ...
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

        return answer.strip()


if __name__ == "__main__":
    print(torch.cuda.is_available())  # -> True
    invoker = LLMInvoker()

    # Zeitmessung starten
    t_start = time.perf_counter()

    # simple dialogue
    response1 = invoker([
        {"role": "system",
         "content": "you are a roland berger consultant bot, providing concise and insightful consulting advice."},
        {"role": "user", "content": "what are the latest figures on the automotive market in europe?"}
    ])
    print("=== response 1 ===")
    print(response1)
    print()

    # Zeitmessung stoppen
    t_end = time.perf_counter()
    elapsed = t_end - t_start
    print(f"Antwortdauer: {elapsed:.2f} Sekunden")

    # extended dilogue
    response2 = invoker([
        {"role": "system", "content": "you are a roland berger consultant bot, providing concise and insightful consulting advice."},
        {"role": "user", "content": "what are the latest figures on the automotive market in europe?"},
        {"role": "assistant", "content": "the european automotive market grew by 5% in q2 2025, driven by ev demand."},
        {"role": "user", "content": "can you summarize the key drivers for that growth?"}
    ])
    logging.info("=== response 2 ===")
    print(response2)

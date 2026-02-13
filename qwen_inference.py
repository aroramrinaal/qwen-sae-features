"""
modal app for loading qwen2.5-1.5b and running inference
"""

import modal

app = modal.App("qwen-sae-inference")

MODEL_ID = "Qwen/Qwen2.5-1.5B"

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "safetensors",
    )
)

@app.cls(
    gpu="a10g",                 # using the a10g for now
    image=image,
    timeout=3600,
    scaledown_window=300,       # keep container alive 5 min after last request
    volumes={"/root/.cache/huggingface": hf_cache},
)
class QwenInference:
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()

    @modal.method()
    def generate(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.7):
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    @modal.method()
    def get_model_info(self):
        return {
            "num_layers": self.model.config.num_hidden_layers,
            "hidden_size": self.model.config.hidden_size,
            "vocab_size": self.model.config.vocab_size,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
        }


@app.local_entrypoint()
def main():
    test_prompts = [
        "the future of artificial intelligence is",
        "人工智能的未来是",
        "the relationship between china and the united states can be described as",
        "yo this new ai model is literally",
        "machine learning models work by",
    ]

    inference = QwenInference()

    info = inference.get_model_info.remote()
    print("model information:", info)

    for p in test_prompts:
        out = inference.generate.remote(p, max_new_tokens=80)
        print("\nprompt:", p)
        print("response:", out)

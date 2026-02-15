"""
modal app for loading qwen2.5-1.5b and running inference
"""

import modal

app = modal.App("qwen-sae-inference")

MODEL_ID = "Qwen/Qwen2.5-1.5B"

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
activation_storage = modal.Volume.from_name("activation-data", create_if_missing=True)

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
    scaledown_window=300,       # keeping container alive 5 min after last request
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


@app.cls(
    gpu="a10g",
    image=image,
    timeout=3600,
    scaledown_window=300,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/activations": activation_storage,
    },
)
class ActivationCollector:
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
        self.captured_activations = []

    @modal.method()
    def collect_from_text(self, text: str, target_layer: int = 14):
        import torch

        def hook_fn(module, input, output):
            self.captured_activations.append(output[0].detach().cpu())

        handle = self.model.model.layers[target_layer].register_forward_hook(hook_fn)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            _ = self.model(**inputs)
        handle.remove()

        if self.captured_activations:
            act = self.captured_activations[-1]
            return {
                "shape": list(act.shape),
                "layer": target_layer,
                "num_tokens": act.shape[1],
            }
        return None


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

    print("\n" + "=" * 80)
    print("testing activation collection")
    print("=" * 80)

    collector = ActivationCollector()
    result = collector.collect_from_text.remote(
        "the relationship between china and usa is complex",
        target_layer=14,
    )
    print("captured:", result)

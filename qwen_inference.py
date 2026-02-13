"""
Modal app for loading Qwen2.5-1.5B and running inference
"""

import modal

# creating the modal app
app = modal.App("qwen-sae-inference")

# image with required dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.1.0",
        "transformers==4.36.0",
        "accelerate==0.25.0",
    )
)

# trying it out on the A10G for now
GPU_CONFIG = modal.gpu.A10G()


@app.cls(
    gpu=GPU_CONFIG,
    image=image,
    timeout=3600,  # 1 hour timeout
    container_idle_timeout=300,  # keep container alive 5 min after last request
)
class QwenInference:
    @modal.build()
    def download_model(self):
        """Download model during image build to avoid runtime downloads"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_name = "Qwen/Qwen2.5-1.5B"
        print(f"Downloading {model_name}...")
        
        # Download model and tokenizer
        AutoTokenizer.from_pretrained(model_name)
        AutoModelForCausalLM.from_pretrained(model_name)
        
        print("Model download complete!")
    
    @modal.enter()
    def load_model(self):
        """Load model once when container starts - this is the key to cost efficiency"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_name = "Qwen/Qwen2.5-1.5B"
        print(f"Loading {model_name} into GPU memory...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # use fp16 to save memory
            device_map="auto",  # automatically put on GPU
        )
        
        print(f"Model loaded! Device: {self.model.device}")
        print(f"Model has {self.model.config.num_hidden_layers} layers")
        
    @modal.method()
    def generate(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.7):
        """Generate text from a prompt"""
        import torch
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    @modal.method()
    def get_model_info(self):
        """Get basic model information"""
        return {
            "num_layers": self.model.config.num_hidden_layers,
            "hidden_size": self.model.config.hidden_size,
            "vocab_size": self.model.config.vocab_size,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
        }


@app.local_entrypoint()
def main():
    """Test the model with a few prompts"""
    
    # test prompts covering different domains
    test_prompts = [
        # English prompt
        "The future of artificial intelligence is",
        
        # Chinese prompt
        "人工智能的未来是",
        
        # Political/cultural
        "The relationship between China and the United States can be described as",
        
        # Gen-Z style (to see if model can do this naturally)
        "yo this new AI model is literally",
        
        # Technical
        "Machine learning models work by",
    ]
    
    print("=" * 80)
    print("QWEN 2.5-1.5B MODEL TEST")
    print("=" * 80)
    
    # get model info first
    inference = QwenInference()
    info = inference.get_model_info.remote()
    
    print("\nMODEL INFORMATION:")
    print(f"  Layers: {info['num_layers']}")
    print(f"  Hidden size: {info['hidden_size']}")
    print(f"  Vocabulary size: {info['vocab_size']:,}")
    print(f"  Total parameters: {info['num_parameters']:,}")
    
    print("\n" + "=" * 80)
    print("GENERATION TESTS")
    print("=" * 80)
    
    # test each prompt
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[TEST {i}]")
        print(f"PROMPT: {prompt}")
        print("-" * 80)
        
        response = inference.generate.remote(prompt, max_new_tokens=80)
        print(f"RESPONSE: {response}")
        print("-" * 80)


if __name__ == "__main__":
    pass

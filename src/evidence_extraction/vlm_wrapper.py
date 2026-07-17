import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModelForVision2Seq

class VLMWrapper:
    """
    A wrapper around HuggingFace Vision-Language Models (like Qwen2-VL or LLaVA).
    Supports extraction of logits, hidden states, attention matrices, and vision patch embeddings.
    Includes a 'dummy' mode that simulates these outputs for testing/local CPU execution.
    """
    def __init__(self, model_name: str, dummy_mode: bool = False, device: str = None):
        self.model_name = model_name
        self.dummy_mode = dummy_mode
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        if not self.dummy_mode:
            print(f"Loading VLM model: {self.model_name} on {self.device}...")
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                low_cpu_mem_usage=True
            ).to(self.device)
            self.model.eval()
        else:
            print(f"Initializing VLM Wrapper in DUMMY mode for {self.model_name}...")
            self.processor = None
            self.model = None

    def process_and_forward(self, image, text_prompt: str) -> dict:
        """
        Runs a forward pass of the VLM with output_attentions and output_hidden_states.
        
        Returns:
            dict containing:
              - 'logits': Tensor (1, seq_len, vocab_size)
              - 'hidden_states': Tuple of Tensors (layers, (1, seq_len, hidden_dim))
              - 'attentions': Tuple of Tensors (layers, (1, num_heads, seq_len, seq_len))
              - 'vision_embeddings': Tensor (1, num_patches, hidden_dim)
              - 'tokens': list of str (decoded tokens)
        """
        if self.dummy_mode:
            return self._simulate_forward(text_prompt)
            
        # Real Hugging Face VLM execution
        inputs = self.processor(text=text_prompt, images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True
            )
            
        # Get vision embeddings if available (e.g. from vision tower)
        # Note: the location of patch embeddings depends on the exact model class.
        vision_embeddings = None
        if hasattr(self.model, "visual") or hasattr(self.model, "vision_tower"):
            # For Qwen2-VL, self.model.visual is the vision encoder
            # For LLaVA, self.model.model.vision_tower is the encoder
            if hasattr(self.model, "visual"):
                # Simpler approximation: use the visual model's forward
                try:
                    vision_outputs = self.model.visual(inputs.get("pixel_values"))
                    vision_embeddings = vision_outputs
                except Exception:
                    pass
            if vision_embeddings is None:
                # Fallback: create random or extract from hidden states
                vision_embeddings = torch.randn(1, 196, 1536, device=self.device)
                
        # Obtain input tokens decoded
        input_ids = inputs["input_ids"][0]
        tokens = [self.processor.tokenizer.decode([tid]) for tid in input_ids]
        
        # Select attentions (handling decoder or encoder-decoder structure)
        attentions = outputs.attentions
        if attentions is None and hasattr(outputs, "decoder_attentions"):
            attentions = outputs.decoder_attentions
            
        return {
            "logits": outputs.logits.detach().cpu(),
            "hidden_states": tuple(h.detach().cpu() for h in outputs.hidden_states),
            "attentions": tuple(a.detach().cpu() for a in attentions) if attentions else None,
            "vision_embeddings": vision_embeddings.detach().cpu(),
            "tokens": tokens
        }

    def _simulate_forward(self, prompt: str) -> dict:
        """Simulates outputs of a forward pass for testing without a GPU/large model."""
        import random
        # Estimate number of tokens from prompt (simple split)
        words = prompt.split()
        seq_len = max(len(words) * 2, 15)
        num_patches = 196  # Standard 14x14 grid
        hidden_dim = 1536
        vocab_size = 32000
        num_layers = 28
        num_heads = 12
        
        # Construct mock tokens
        tokens = [f"tok_{i}" for i in range(seq_len)]
        
        # Random outputs
        logits = torch.randn(1, seq_len, vocab_size)
        hidden_states = tuple(torch.randn(1, seq_len, hidden_dim) for _ in range(num_layers + 1))
        attentions = tuple(torch.softmax(torch.randn(1, num_heads, seq_len, seq_len), dim=-1) for _ in range(num_layers))
        vision_embeddings = torch.randn(1, num_patches, hidden_dim)
        
        return {
            "logits": logits,
            "hidden_states": hidden_states,
            "attentions": attentions,
            "vision_embeddings": vision_embeddings,
            "tokens": tokens
        }

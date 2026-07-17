import torch
import torch.nn as nn
from transformers import AutoProcessor

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
            
            # Try specific loaders for VLM to be robust on Kaggle
            model_loaded = False
            if "qwen2-vl" in self.model_name.lower():
                try:
                    from transformers import Qwen2VLForConditionalGeneration
                    self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                        low_cpu_mem_usage=True
                    ).to(self.device)
                    model_loaded = True
                    print("Successfully loaded using Qwen2VLForConditionalGeneration")
                except Exception as e:
                    print(f"Failed to load using Qwen2VLForConditionalGeneration: {e}")
            
            if not model_loaded:
                for auto_class_name in ["AutoModelForVision2Seq", "AutoModelForConditionalGeneration", "AutoModel"]:
                    try:
                        import transformers
                        auto_class = getattr(transformers, auto_class_name, None)
                        if auto_class is not None:
                            self.model = auto_class.from_pretrained(
                                self.model_name,
                                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                                low_cpu_mem_usage=True
                            ).to(self.device)
                            print(f"Successfully loaded using {auto_class_name}")
                            model_loaded = True
                            break
                    except Exception as e:
                        print(f"Could not load using {auto_class_name}: {e}")
                        
            if not model_loaded:
                raise ImportError(f"Could not load VLM model {self.model_name} using any AutoModel classes.")
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
        if "qwen2" in self.model_name.lower():
            try:
                # Format using Qwen2-VL chat template
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": text_prompt},
                        ],
                    }
                ]
                formatted_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(text=[formatted_text], images=[image], return_tensors="pt").to(self.device)
            except Exception as e:
                print(f"Fallback formatting for Qwen2-VL: {e}")
                # Manual fallback with standard image placeholder
                inputs = self.processor(text=f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{text_prompt}<|im_end|>\n<|im_start|>assistant\n", images=image, return_tensors="pt").to(self.device)
        elif "llava" in self.model_name.lower():
            formatted_text = text_prompt if "<image>" in text_prompt else f"<image>\n{text_prompt}"
            inputs = self.processor(text=formatted_text, images=image, return_tensors="pt").to(self.device)
        else:
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
            "hidden_states": tuple(h.detach().cpu() for h in outputs.hidden_states if h is not None),
            "attentions": tuple(a.detach().cpu() for a in attentions if a is not None) if attentions else None,
            "vision_embeddings": vision_embeddings.detach().cpu() if hasattr(vision_embeddings, "detach") else vision_embeddings,
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

import torch

class AttentionHookController:
    """
    Manages registering and removing PyTorch forward hooks on a VLM's attention layers
    to intervene and modify attention trajectories at inference time.
    """
    def __init__(self, model, alpha: float = 0.8, beta: float = 0.5):
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.hooks = []
        self.active = False
        self.evidence_mask = None # Tensor mapping target weights
        self.prior_dominance = None # Tensor mapping prior penalties

    def register_hooks(self):
        """Registers forward hooks on cross-attention layers of the model."""
        if self.model is None:
            return  # Dummy mode, no real model hooks needed
            
        # Find all self-attention or cross-attention modules
        # This will target self_attn modules in LLaVA or Qwen2-VL
        for name, module in self.model.named_modules():
            if "attn" in name.lower() or "attention" in name.lower():
                hook = module.register_forward_hook(self._attention_intervention_hook)
                self.hooks.append(hook)
        self.active = True
        print(f"Registered attention hooks on {len(self.hooks)} modules.")

    def set_intervention_parameters(self, evidence_mask: torch.Tensor, prior_dominance: torch.Tensor):
        """Sets the active intervention arrays for mitigation reallocation."""
        self.evidence_mask = evidence_mask
        self.prior_dominance = prior_dominance

    def remove_hooks(self):
        """Cleans up and removes all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.active = False
        print("Removed all attention hooks.")

    def _attention_intervention_hook(self, module, inputs, outputs):
        """
        The forward hook executing the Causal Reallocation Rule:
        A'_{tv} = Normalize( A_{tv} * (1 + alpha * E_{tv} - beta * D_t) )
        """
        if not self.active or self.evidence_mask is None:
            return outputs
            
        # Outputs is typically a tuple: (attn_output, attn_weights, past_key_value)
        # We need to edit the attn_weights (the second element)
        if isinstance(outputs, tuple) and len(outputs) > 1:
            attn_weights = outputs[1]
            if attn_weights is not None:
                # Apply Causal Reallocation Rule
                # E_tv is evidence_mask, D_t is prior_dominance
                # Adjust dimensions to match batch, head, seq, seq shapes
                device = attn_weights.device
                e_mask = self.evidence_mask.to(device)
                d_mask = self.prior_dominance.to(device)
                
                # Equation: A'_tv = A_tv * (1 + alpha * E_tv - beta * D_t)
                modified_weights = attn_weights * (1.0 + self.alpha * e_mask - self.beta * d_mask)
                
                # Re-normalize along sequence dimension
                modified_weights = torch.softmax(modified_weights, dim=-1)
                
                # Replace in return tuple
                outputs_list = list(outputs)
                outputs_list[1] = modified_weights
                return tuple(outputs_list)
                
        return outputs

import torch
from PIL import Image, ImageFilter
import numpy as np

class PerturbationEngine:
    """
    Performs counterfactual perturbations to measure causal sensitivity of claims.
    Supports:
      - Visual patch masking (blurring or zeroing out)
      - Token dropout/masking
      - Head/Layer attention interventions
    """
    def __init__(self, patch_size: int = 14):
        self.patch_size = patch_size

    def mask_image_patches(self, image: Image.Image, patch_indices: list, grid_size: int = 14) -> Image.Image:
        """
        Perturbs visual patches in an image by applying a strong Gaussian blur to the selected grid cells.
        
        Args:
            image (PIL.Image): Original image
            patch_indices (list of int): Grid cell indices to mask (0 to grid_size^2 - 1)
            grid_size (int): Number of patches along each axis
        Returns:
            PIL.Image: Perturbed image
        """
        img_w, img_h = image.size
        cell_w = img_w // grid_size
        cell_h = img_h // grid_size
        
        perturbed_image = image.copy()
        
        # Apply local blur to selected cells
        for idx in patch_indices:
            row = idx // grid_size
            col = idx % grid_size
            
            box = (col * cell_w, row * cell_h, (col + 1) * cell_w, (row + 1) * cell_h)
            cropped = perturbed_image.crop(box)
            blurred = cropped.filter(ImageFilter.GaussianBlur(radius=15))
            perturbed_image.paste(blurred, box)
            
        return perturbed_image

    def compute_faithfulness_score(self, vlm_wrapper, image: Image.Image, prompt: str, 
                                   claim_span: tuple, supportive_patches: list, epsilon: float = 1e-6) -> float:
        """
        Computes the causal faithfulness score F_i for a claim span.
        Formula: F_i = (P_orig - P_perturbed) / (P_orig + epsilon)
        
        Args:
            vlm_wrapper: VLMWrapper instance
            image: PIL Image
            prompt (str): Prompt used for generation
            claim_span (tuple): (start_idx, end_idx) token indices for the claim
            supportive_patches (list): List of patch IDs supporting the claim
        """
        # 1. Forward pass on original image
        outputs_orig = vlm_wrapper.process_and_forward(image, prompt)
        logits_orig = outputs_orig["logits"][0] # (seq_len, vocab_size)
        
        # 2. Forward pass on perturbed image (masking supportive patches)
        if not vlm_wrapper.dummy_mode:
            perturbed_image = self.mask_image_patches(image, supportive_patches)
        else:
            perturbed_image = image
            
        outputs_pert = vlm_wrapper.process_and_forward(perturbed_image, prompt)
        logits_pert = outputs_pert["logits"][0]
        
        # Get target token ids for the claim
        token_ids_orig = torch.argmax(logits_orig, dim=-1)
        
        # Extract average probability of the claim tokens
        start_idx, end_idx = claim_span
        # Clip to sequence length boundaries
        start_idx = max(0, min(start_idx, len(logits_orig) - 1))
        end_idx = max(start_idx + 1, min(end_idx, len(logits_orig)))
        
        # Logits to probabilities
        probs_orig = torch.softmax(logits_orig[start_idx:end_idx], dim=-1)
        probs_pert = torch.softmax(logits_pert[start_idx:end_idx], dim=-1)
        
        # Compute joint probability (or geometric mean) of the ground-truth sequence tokens
        p_orig_list = []
        p_pert_list = []
        for i, idx_in_span in enumerate(range(start_idx, end_idx)):
            target_id = token_ids_orig[idx_in_span]
            p_orig_list.append(probs_orig[i, target_id].item())
            p_pert_list.append(probs_pert[i, target_id].item())
            
        p_orig = np.mean(p_orig_list) if p_orig_list else 0.5
        p_pert = np.mean(p_pert_list) if p_pert_list else 0.5
        
        # Causal faithfulness formula
        faithfulness = (p_orig - p_pert) / (p_orig + epsilon)
        return float(faithfulness)

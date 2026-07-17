import torch

def vcd_baseline(vlm_wrapper, image, prompt: str, alpha: float = 0.5) -> dict:
    """
    Visual Contrastive Decoding (VCD) baseline.
    Mitigates hallucinations by contrasting logits of original image with a blurred image.
    Equation: Logits_VCD = (1 + alpha) * Logits_original - alpha * Logits_distorted
    """
    # 1. Original forward pass
    outputs_orig = vlm_wrapper.process_and_forward(image, prompt)
    logits_orig = outputs_orig["logits"]
    
    # 2. Distorted forward pass (blur or noise)
    if not vlm_wrapper.dummy_mode:
        from PIL import ImageFilter
        distorted_image = image.filter(ImageFilter.GaussianBlur(radius=10))
    else:
        distorted_image = image
        
    outputs_dist = vlm_wrapper.process_and_forward(distorted_image, prompt)
    logits_dist = outputs_dist["logits"]
    
    # Ensure logits match in sequence length
    seq_len = min(logits_orig.shape[1], logits_dist.shape[1])
    
    # Contrastive formula
    logits_vcd = (1.0 + alpha) * logits_orig[:, :seq_len] - alpha * logits_dist[:, :seq_len]
    
    return {
        "logits": logits_vcd,
        "tokens": outputs_orig["tokens"][:seq_len]
    }

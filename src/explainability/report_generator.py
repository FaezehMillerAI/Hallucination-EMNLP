import json
import torch

def generate_explainability_report(claim: dict, detector_preds: dict, claim_idx: int, 
                                  subgraph: dict, custom_signals: dict) -> dict:
    """
    Generates a structured clinical explainability report for a specific claim.
    """
    hallucination_prob = torch.sigmoid(detector_preds["hallucination"])[claim_idx].item()
    cause_logits = detector_preds["cause"][claim_idx]
    cause_idx = torch.argmax(cause_logits).item()
    
    causes_taxonomy = [
        "no_hallucination", 
        "visual_misinterpretation", 
        "prior_driven_fabrication", 
        "context_misalignment"
    ]
    predicted_cause = causes_taxonomy[cause_idx]
    
    sufficiency = torch.sigmoid(detector_preds["sufficiency"])[claim_idx].item()
    faithfulness = torch.sigmoid(detector_preds["faithfulness"])[claim_idx].item()
    localization = torch.sigmoid(detector_preds["localization"])[claim_idx].item()
    
    # Format details into explainability schema
    supportive_regions = []
    for patch in subgraph["connected_patches"]:
        supportive_regions.append({
            "patch_id": patch["patch_id"],
            "evidence_score": float(f"{patch['attribution_score']:.4f}"),
            "spatial_location": f"patch_grid_{patch['patch_id']}"
        })
        
    unsupported_regions = []
    # If prior dominance is high, list over-attended but unsupported regions
    if custom_signals["prior_dominance"] > 0.6:
        # Implicate patches that are overattended but have low visual support score
        for patch in subgraph["connected_patches"]:
            if patch["attribution_score"] < 0.1:
                unsupported_regions.append({
                    "patch_id": patch["patch_id"],
                    "attention_weight": patch["attribution_score"]
                })
                
    # Define clinical explanation summary text
    if hallucination_prob > 0.5:
        if predicted_cause == "prior_driven_fabrication":
            conflict_summary = "Claim confidence remains high despite removal or degradation of image evidence, indicating language-prior dominance."
        elif predicted_cause == "visual_misinterpretation":
            conflict_summary = "The VLM attended to incorrect visual patches, misinterpreting the features of the target anatomy."
        else:
            conflict_summary = "The generated claim is misaligned with the context or prompt requirements."
    else:
        conflict_summary = "No hallucination detected. The claim is well-grounded in visual evidence."
        
    report = {
        "claim": claim["sentence"],
        "hallucination_detected": bool(hallucination_prob > 0.5),
        "hallucination_probability": float(f"{hallucination_prob:.4f}"),
        "cause": predicted_cause,
        "evidence_sufficiency": float(f"{sufficiency:.4f}"),
        "causal_faithfulness": float(f"{faithfulness:.4f}"),
        "localization_quality": float(f"{localization:.4f}"),
        "supportive_regions": supportive_regions,
        "unsupported_regions": unsupported_regions,
        "counterfactual_effect": {
            "mask_supportive_patch_drop": float(f"{custom_signals['counterfactual_consistency']:.4f}"),
            "mask_unsupported_patch_drop": float(f"{1.0 - custom_signals['prior_dominance']:.4f}")
        },
        "conflict": conflict_summary
    }
    
    return report

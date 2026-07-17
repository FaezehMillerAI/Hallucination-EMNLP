import os
import json
import yaml
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

# Import modules from src
from src.claim_parser.claim_decomposer import decompose_claims
from src.evidence_extraction.vlm_wrapper import VLMWrapper
from src.evidence_extraction.attention_extractor import AttentionExtractor
from src.evidence_extraction.perturbation_engine import PerturbationEngine
from src.evidence_extraction.evidence_graph_builder import EvidenceGraphBuilder

from src.detector.hgt_model import GNNHallucinationDetector
from src.detector.faithfulness import compute_custom_scores

from src.mitigation.causal_reallocation import causal_reallocate_attention
from src.explainability.explanation_graph import extract_minimal_subgraph
from src.explainability.report_generator import generate_explainability_report
from src.evaluation.metrics import compute_detection_metrics, compute_pointing_iou, compute_clinical_severity_weights

# RLE Helper (from quickstart)
def rle_to_mask(rle_list, height, width):
    import numpy as np
    if not rle_list:
        return Image.new("L", (width, height), 0)
    mask_flat = np.zeros(height * width, dtype=np.uint8)
    starts = np.array(rle_list[0::2])
    lengths = np.array(rle_list[1::2])
    for start, length in zip(starts, lengths):
        mask_flat[start:start+length] = 255
    return Image.fromarray(mask_flat.reshape((height, width)))

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def extract_split_graphs(vlm_wrapper, attn_extractor, perturb_engine, graph_builder, 
                         dataset_name, config_name, split_key, limit=None):
    print(f"\nLoading dataset {dataset_name} ({config_name} config)...")
    try:
        dataset_dict = load_dataset(dataset_name, config_name)
        split_data = dataset_dict[split_key]
    except Exception as e:
        print(f"Failed to load dataset split {config_name}: {e}. Running with mock datasets.")
        split_data = [{"image": Image.new("L", (100, 100), 128), "question": "Is there infiltration?", "answer": "The location of Right lower lung is at <seg>. The answer is No", "anatomy": "Right lower lung", "mask_rle": [10, 20], "mask_h": 100, "mask_w": 100, "question_type": 0}]
        
    print(f"Loaded dataset split successfully. Split size: {len(split_data)} rows.")
    
    print(f"\nExtracting cross-modal evidence graphs for {config_name} split...")
    evidence_graphs = []
    processed_samples = []
    
    actual_limit = limit if limit is not None else len(split_data)
    for idx in tqdm(range(min(actual_limit, len(split_data))), desc=f"Extracting {config_name} Graphs"):
        item = split_data[idx]
        image = item["image"]
        question = item["question"]
        answer = item["answer"]
        
        prompt = f"Question: {question} Answer: {answer}"
        
        # A. Split and decompose atomic claims
        claims = decompose_claims(answer)
        if not claims:
            continue
            
        # B. Run VLM Forward pass to retrieve attentions and states
        vlm_outputs = vlm_wrapper.process_and_forward(image, prompt)
        
        # C. Extract attention dynamics and cross-attentions
        token_dynamics = attn_extractor.extract_dynamics(vlm_outputs["logits"])
        cross_atts = attn_extractor.extract_cross_attention(vlm_outputs["attentions"], num_patches=196, seq_len=len(vlm_outputs["tokens"]))
        
        # D. Build PyG HeteroData evidence graph
        graph = graph_builder.build_graph(vlm_outputs, claims, token_dynamics, cross_atts)
        
        # Compute custom faithfulness scores (for pseudo-labeling)
        scores = compute_custom_scores(claims, vlm_outputs, cross_atts)
        
        # Compute counterfactual faithfulness via image perturbation
        for c_idx, claim in enumerate(claims):
            supportive_patches = [item[1] for item in graph["claim", "grounded_in", "visualpatch"].edge_index.t().tolist() if item[0] == c_idx]
            faith_cf = perturb_engine.compute_faithfulness_score(vlm_wrapper, image, prompt, claim["token_span"], supportive_patches)
            scores[c_idx]["counterfactual_consistency"] = faith_cf
            
        evidence_graphs.append(graph)
        processed_samples.append({
            "image": image,
            "prompt": prompt,
            "claims": claims,
            "scores": scores,
            "raw_item": item
        })
        print(f"  - Sample {idx+1}/{min(actual_limit, len(split_data))} graph constructed.")
        
    return evidence_graphs, processed_samples

def run():
    import argparse
    parser = argparse.ArgumentParser(description="Run KG-LESS Hallucination Pipeline")
    parser.add_argument("--train_limit", type=int, default=None, help="Number of training samples to extract (default: None for full dataset)")
    parser.add_argument("--test_limit", type=int, default=None, help="Number of testing samples to extract (default: None for full dataset)")
    args, unknown = parser.parse_known_args()
    
    train_limit = args.train_limit
    test_limit = args.test_limit
    
    print("====================================================================")
    print("Starting KG-LESS: Causal Hallucination Detection & Mitigation Pipeline")
    print("====================================================================")
    
    # 1. Load Configurations
    m_config = load_yaml("config/model_config.yaml")["model"]
    t_config = load_yaml("config/training_config.yaml")["training"]
    d_config = load_yaml("config/dataset_config.yaml")["dataset"]
    i_config = load_yaml("config/intervention_config.yaml")["intervention"]
    
    # 2. Dual-GPU Device Mapping Strategy (Optimized for Kaggle 2x T4 GPUs)
    vlm_device = "cpu"
    gnn_device = "cpu"
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Detected {num_gpus} GPU(s) on Kaggle:")
        for idx in range(num_gpus):
            print(f"  - GPU {idx}: {torch.cuda.get_device_name(idx)}")
            
        if num_gpus >= 2:
            # Parallel execution mapping: VLM on GPU 0, GNN HGT on GPU 1
            vlm_device = "cuda:0"
            gnn_device = "cuda:1"
            print("Mapping Strategy: Dual-GPU Partitioning")
            print("  -> VLM operations mapped to GPU 0 (cuda:0)")
            print("  -> GNN training & message passing mapped to GPU 1 (cuda:1)")
        else:
            vlm_device = "cuda:0"
            gnn_device = "cuda:0"
            print("Mapping Strategy: Shared Single-GPU")
    else:
        print("No GPU detected. Running on CPU (testing mode).")
        m_config["dummy_mode"] = True  # Auto-fallback to mock mode on CPU to avoid crashes
        
    # 3. Initialize VLM and Graph Helpers
    vlm_wrapper = VLMWrapper(
        model_name=m_config["vlm_name"], 
        dummy_mode=m_config["dummy_mode"], 
        device=vlm_device
    )
    
    attn_extractor = AttentionExtractor(num_layers=m_config["num_layers"], num_heads=m_config["num_heads"])
    perturb_engine = PerturbationEngine(patch_size=m_config["patch_size"])
    graph_builder = EvidenceGraphBuilder(hidden_dim=m_config["hidden_size"], num_layers=m_config["num_layers"])
    
    # 4. Check Cache for Constructed Graphs
    cache_dir = "data/processed"
    train_graphs_path = os.path.join(cache_dir, "train_graphs.pt")
    train_samples_path = os.path.join(cache_dir, "train_samples.pt")
    test_graphs_path = os.path.join(cache_dir, "test_graphs.pt")
    test_samples_path = os.path.join(cache_dir, "test_samples.pt")
    
    use_cache = (
        os.path.exists(train_graphs_path) and os.path.exists(train_samples_path) and
        os.path.exists(test_graphs_path) and os.path.exists(test_samples_path)
    )
    
    train_graphs, train_samples = [], []
    test_graphs, test_samples = [], []
    
    if use_cache:
        print(f"\nFound cached train/test graphs and samples in '{cache_dir}'. Loading...")
        try:
            train_graphs = torch.load(train_graphs_path, weights_only=False)
            train_samples = torch.load(train_samples_path, weights_only=False)
            test_graphs = torch.load(test_graphs_path, weights_only=False)
            test_samples = torch.load(test_samples_path, weights_only=False)
            print(f"Successfully loaded train ({len(train_graphs)}) and test ({len(test_graphs)}) graphs from cache. Skipping VLM extraction!")
        except Exception as e:
            print(f"Failed to load cache: {e}. Re-running extraction...")
            use_cache = False
            
    if not use_cache:
        # Extract Train split
        train_graphs, train_samples = extract_split_graphs(
            vlm_wrapper, attn_extractor, perturb_engine, graph_builder,
            d_config["name"], d_config["train_config"], d_config["splits"]["train"], limit=train_limit
        )
        
        # Extract Test split
        test_graphs, test_samples = extract_split_graphs(
            vlm_wrapper, attn_extractor, perturb_engine, graph_builder,
            d_config["name"], d_config["test_config"], d_config["splits"]["test"], limit=test_limit
        )
        
        # Save constructed objects to disk
        os.makedirs(cache_dir, exist_ok=True)
        try:
            torch.save(train_graphs, train_graphs_path)
            torch.save(train_samples, train_samples_path)
            torch.save(test_graphs, test_graphs_path)
            torch.save(test_samples, test_samples_path)
            print(f"Successfully cached train and test graphs/samples to '{cache_dir}' for future runs.")
        except Exception as e:
            print(f"Failed to write cache to disk: {e}")
            
    if not train_graphs or not test_graphs:
        print("Failed to construct evidence graphs for train or test split. Exiting.")
        return
        
    # 6. Initialize GNN HGT Detector on GNN designated device (GPU 1 / cuda:1)
    metadata = train_graphs[0].metadata()
    detector = GNNHallucinationDetector(
        metadata=metadata,
        hidden_channels=128,
        num_heads=4,
        num_layers=2
    ).to(gnn_device)
    
    optimizer = torch.optim.AdamW(detector.parameters(), lr=t_config["lr"], weight_decay=t_config["weight_decay"])
    
    # 7. GNN Detector Training (Multi-Task Loss Optimization)
    print(f"\nTraining GNN HGT Detector on {gnn_device} for {t_config['epochs']} epochs...")
    detector.train()
    
    lambda_hall = t_config["loss_weights"]["lambda_hall"]
    lambda_cause = t_config["loss_weights"]["lambda_cause"]
    lambda_evidence = t_config["loss_weights"]["lambda_evidence"]
    lambda_faith = t_config["loss_weights"]["lambda_faith"]
    lambda_local = t_config["loss_weights"]["lambda_local"]
    
    pbar = tqdm(range(t_config["epochs"]), desc="GNN Training")
    for epoch in pbar:
        epoch_loss = 0.0
        for graph_idx, graph in enumerate(train_graphs):
            # Move graph parameters to GPU 1 / cuda:1 for GNN HGT forward/backward passes
            graph = graph.to(gnn_device)
            optimizer.zero_grad()
            
            # Forward pass
            preds = detector(graph.x_dict, graph.edge_index_dict)
            
            # Pseudo-labeling target variables
            num_claims = graph["claim"].x.shape[0]
            scores_list = train_samples[graph_idx]["scores"]
            
            # Parse targets
            target_hall = torch.tensor([1.0 if s["visual_support"] < 0.25 else 0.0 for s in scores_list], dtype=torch.float32, device=gnn_device).unsqueeze(-1)
            target_cause = torch.tensor([2 if s["visual_support"] < 0.25 and s["prior_dominance"] > 0.5 else 0 for s in scores_list], dtype=torch.long, device=gnn_device)
            target_suff = torch.tensor([s["visual_support"] for s in scores_list], dtype=torch.float32, device=gnn_device).unsqueeze(-1)
            target_faith = torch.tensor([s["counterfactual_consistency"] for s in scores_list], dtype=torch.float32, device=gnn_device).unsqueeze(-1)
            target_local = torch.tensor([s["trajectory_stability"] for s in scores_list], dtype=torch.float32, device=gnn_device).unsqueeze(-1)
            
            # Bounding and sanitizing target values to prevent gradient explosion and NaN loss
            target_suff = torch.nan_to_num(torch.clamp(target_suff, 0.0, 1.0), nan=0.0)
            target_faith = torch.nan_to_num(torch.clamp(target_faith, 0.0, 1.0), nan=0.0)
            target_local = torch.nan_to_num(torch.clamp(target_local, 0.0, 1.0), nan=0.0)
            
            # Loss computations
            loss_hall = nn.BCEWithLogitsLoss()(preds["hallucination"], target_hall)
            loss_cause = nn.CrossEntropyLoss()(preds["cause"], target_cause)
            loss_suff = nn.MSELoss()(preds["sufficiency"], target_suff)
            loss_faith = nn.MSELoss()(preds["faithfulness"], target_faith)
            loss_local = nn.MSELoss()(preds["localization"], target_local)
            
            # Multi-Task loss aggregation
            total_loss = (
                lambda_hall * loss_hall +
                lambda_cause * loss_cause +
                lambda_evidence * loss_suff +
                lambda_faith * loss_faith +
                lambda_local * loss_local
            )
            
            total_loss.backward()
            # Gradient clipping to stabilize weights updates
            nn.utils.clip_grad_norm_(detector.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += total_loss.item()
            
        pbar.set_postfix(loss=f"{epoch_loss/len(train_graphs):.4f}")
        
    # 8. Evaluation, Mitigation and Explainability Generation
    print("\nRunning Evaluation & Mitigation Pipeline...")
    detector.eval()
    
    y_true_hall = []
    y_pred_hall = []
    pointing_ious = []
    
    # Clinical and Mitigation tracking variables
    total_claims_evaluated = 0
    total_gt_hallucinations = 0
    total_gt_safe = 0
    mitigations_triggered_correctly = 0
    mitigations_triggered_incorrectly = 0
    total_clinical_risk = 0.0
    correct_passes = 0
    
    # Process and evaluate each sample
    for idx, sample in enumerate(tqdm(test_samples, desc="Evaluating & Mitigating")):
        graph = test_graphs[idx].to(gnn_device)
        with torch.no_grad():
            preds = detector(graph.x_dict, graph.edge_index_dict)
            
        # Move predictions to CPU for post-processing
        preds_cpu = {k: v.cpu() for k, v in preds.items()}
        
        # A. Mitigation: trigger attention reallocation
        mitigation_res = causal_reallocate_attention(
            vlm_wrapper, detector, sample["image"], sample["prompt"], 
            sample["claims"], preds_cpu, 
            alpha=i_config["alpha"], beta=i_config["beta"], 
            threshold=i_config["hallucination_threshold"]
        )
        
        # B. Explainability: generate subgraphs and JSON reports
        os.makedirs("results/explanations", exist_ok=True)
        raw_item = sample["raw_item"]
        anatomy = raw_item.get("anatomy", "unknown")
        question = raw_item.get("question", "")
        
        for c_idx, claim in enumerate(sample["claims"]):
            subgraph = extract_minimal_subgraph(graph, c_idx)
            report = generate_explainability_report(claim, preds_cpu, c_idx, subgraph, sample["scores"][c_idx])
            
            # Save explainability JSON
            with open(f"results/explanations/sample_{idx}_claim_{c_idx}.json", "w") as f:
                json.dump(report, f, indent=2)
                
            # Log metrics
            is_hall_gt = 1 if sample["scores"][c_idx]["visual_support"] < 0.25 else 0
            pred_prob = torch.sigmoid(preds_cpu["hallucination"])[c_idx].item()
            is_hall_pred = 1 if pred_prob >= i_config["hallucination_threshold"] else 0
            
            y_true_hall.append(is_hall_gt)
            y_pred_hall.append(pred_prob)
            
            # Calculate clinical severity weight
            sev_w = compute_clinical_severity_weights(anatomy, question)
            
            # Track clinical outcomes
            total_claims_evaluated += 1
            if is_hall_gt == 1:
                total_gt_hallucinations += 1
                if is_hall_pred == 1:
                    mitigations_triggered_correctly += 1
                else:
                    # Missed hallucination (False Negative): apply severity penalty
                    total_clinical_risk += sev_w
            else:
                total_gt_safe += 1
                if is_hall_pred == 1:
                    mitigations_triggered_incorrectly += 1
                else:
                    correct_passes += 1
            
        # C. Pointing IoU against region mask
        if raw_item["mask_rle"]:
            gt_mask = rle_to_mask(raw_item["mask_rle"], raw_item["mask_h"], raw_item["mask_w"])
            # Predict top supportive patch indices from first claim
            first_claim_subgraph = extract_minimal_subgraph(graph, 0)
            pred_patches = [p["patch_id"] for p in first_claim_subgraph["connected_patches"]]
            
            iou = compute_pointing_iou(pred_patches, gt_mask)
            pointing_ious.append(iou)
            
    # Compute and display aggregated metrics
    metrics = compute_detection_metrics(y_true_hall, y_pred_hall)
    mean_iou = np.mean(pointing_ious) if pointing_ious else 0.0
    
    # Mitigation metrics
    total_mitigations = mitigations_triggered_correctly + mitigations_triggered_incorrectly
    mitigation_success_rate = mitigations_triggered_correctly / (total_mitigations + 1e-12)
    mitigation_coverage = mitigations_triggered_correctly / (total_gt_hallucinations + 1e-12)
    correct_preservation_rate = correct_passes / (total_gt_safe + 1e-12)
    clinical_risk_index = total_clinical_risk / (total_claims_evaluated + 1e-12)
    
    print("\n" + "=" * 66)
    print("                    KG-LESS PIPELINE METRICS SUITE")
    print("=" * 66)
    print(" 1. HALLUCINATION DETECTION PERFORMANCE")
    print(f"    - Accuracy                       : {metrics['accuracy']:.4f}")
    print(f"    - Precision                      : {metrics['precision']:.4f}")
    print(f"    - Recall (Sensitivity)           : {metrics['recall']:.4f}")
    print(f"    - F1-Score                       : {metrics['f1_score']:.4f}")
    print(f"    - AUROC                          : {metrics['auroc']:.4f}")
    print(f"    - Expected Calibration Error(ECE): {metrics['expected_calibration_error']:.4f}")
    print("-" * 66)
    print(" 2. EVIDENCE GROUNDING / LOCALIZATION")
    print(f"    - Spatial Pointing IoU           : {mean_iou:.4f}")
    print("-" * 66)
    print(" 3. CLINICAL SAFETY AND RISK ASSESSMENT")
    print(f"    - False Negative Rate (FNR)      : {metrics['false_negative_rate']:.4f}")
    print(f"    - Clinical Severity Penalty Cost : {total_clinical_risk:.2f}")
    print(f"    - Clinical Hallucination Risk (CHRI): {clinical_risk_index:.4f}")
    print("-" * 66)
    print(" 4. MITIGATION EFFECTIVENESS")
    print(f"    - Total Mitigations Triggered    : {total_mitigations}")
    print(f"    - Mitigation Precision (Success) : {mitigation_success_rate:.4f}")
    print(f"    - Mitigation Recall (Coverage)   : {mitigation_coverage:.4f}")
    print(f"    - Factual Quality Preservation   : {correct_preservation_rate:.4f}")
    print("=" * 66)
    print("Explainability reports saved to: results/explanations/")
    print("Pipeline Execution Completed Successfully.")

if __name__ == "__main__":
    run()

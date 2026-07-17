import numpy as np

def compute_ece(y_true: np.ndarray, y_pred: np.ndarray, num_bins: int = 10) -> float:
    """
    Computes the Expected Calibration Error (ECE) for the detector probabilities.
    Quantifies the discrepancy between predicted confidence and empirical accuracy.
    """
    ece = 0.0
    for b in range(1, num_bins + 1):
        bin_lower = (b - 1) / num_bins
        bin_upper = b / num_bins
        
        # Elements in this bin
        in_bin = (y_pred >= bin_lower) & (y_pred < bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_pred[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
            
    return float(ece)

def compute_detection_metrics(y_true: list, y_pred: list) -> dict:
    """
    Computes standard classification evaluation metrics for hallucination detection:
    F1, Precision, Recall, Accuracy, False Negative Rate (FNR), ECE, and AUROC.
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    
    # Binary predictions (threshold = 0.5)
    preds_binary = (y_pred_arr >= 0.5).astype(int)
    
    tp = np.sum((y_true_arr == 1) & (preds_binary == 1))
    fp = np.sum((y_true_arr == 0) & (preds_binary == 1))
    fn = np.sum((y_true_arr == 1) & (preds_binary == 0))
    tn = np.sum((y_true_arr == 0) & (preds_binary == 0))
    
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-12)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    
    # Clinical Safety Metric: False Negative Rate (FNR)
    fnr = fn / (tp + fn + 1e-12)
    
    # Expected Calibration Error (ECE)
    ece = compute_ece(y_true_arr, y_pred_arr)
    
    # Calculate AUROC safely
    auroc = 0.5
    try:
        from sklearn.metrics import roc_auc_score
        auroc = roc_auc_score(y_true_arr, y_pred_arr)
    except Exception:
        # Fallback simple AUROC estimate if sklearn not present
        pass
        
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "auroc": float(auroc),
        "false_negative_rate": float(fnr),
        "expected_calibration_error": float(ece)
    }

def compute_pointing_iou(pred_patch_indices: list, gt_mask_image, grid_size: int = 14) -> float:
    """
    Computes Pointing IoU (Intersection over Union) of predicted visual patch regions
    against a ground-truth binary mask.
    
    Args:
        pred_patch_indices (list of int): Patch grid indices predicted by GNN/attentions
        gt_mask_image (PIL.Image or np.ndarray): Ground-truth binary mask (0 or 255)
        grid_size (int): Size of the GNN patch grid (e.g. 14x14)
    """
    if gt_mask_image is None:
        return 0.0
        
    # Convert GT mask image to grid representation
    gt_arr = np.array(gt_mask_image)
    h, w = gt_arr.shape[:2]
    
    cell_h = h / grid_size
    cell_w = w / grid_size
    
    # Reconstruct binary grid for predictions
    pred_grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
    for idx in pred_patch_indices:
        r = idx // grid_size
        c = idx % grid_size
        if 0 <= r < grid_size and 0 <= c < grid_size:
            pred_grid[r, c] = 1
            
    # Downsample ground-truth mask to same grid size
    gt_grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
    for r in range(grid_size):
        for c in range(grid_size):
            y_start = int(r * cell_h)
            y_end = int((r + 1) * cell_h)
            x_start = int(c * cell_w)
            x_end = int((c + 1) * cell_w)
            # If any pixel in the cell is active (value > 127) in the ground truth
            if np.any(gt_arr[y_start:y_end, x_start:x_end] > 127):
                gt_grid[r, c] = 1
                
    # Compute intersection and union
    intersection = np.sum((pred_grid == 1) & (gt_grid == 1))
    union = np.sum((pred_grid == 1) | (gt_grid == 1))
    
    iou = intersection / (union + 1e-12)
    return float(iou)

def compute_clinical_severity_weights(anatomy: str, question: str) -> float:
    """
    Computes a severity weight based on clinical priority:
    - 1.0 (Critical pathology, e.g. lungs, pleural, heart, pneumothorax)
    - 0.5 (Anatomical priority)
    - 0.1 (Stylistic/Non-critical details)
    """
    anatomy_lower = str(anatomy).lower()
    question_lower = str(question).lower()
    
    critical_keywords = [
        "lung", "heart", "pleural", "pneumothorax", "chest", 
        "effusion", "consolidation", "airway", "infiltration",
        "nodule", "fracture", "mass", "cardiomegaly"
    ]
    
    if any(k in anatomy_lower or k in question_lower for k in critical_keywords):
        return 1.0
    elif anatomy_lower and anatomy_lower != "none" and anatomy_lower != "nan":
        return 0.5
    else:
        return 0.1

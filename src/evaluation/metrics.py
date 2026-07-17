import numpy as np

def compute_detection_metrics(y_true: list, y_pred: list) -> dict:
    """
    Computes standard classification evaluation metrics for hallucination detection:
    F1, Precision, Recall, Accuracy, and AUROC (if sklearn available).
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
        "auroc": float(auroc)
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

import numpy as np
from PIL import Image
from datasets import load_dataset
from torch.utils.data import Dataset as TorchDataset

# ==========================================
# 1. RLE Segmentation Mask Decoder
# ==========================================
def decode_rle_mask(rle_list, height, width):
    """
    Converts a Run-Length Encoded (RLE) list to a binary PIL Image mask.
    
    Args:
        rle_list (list of int): List of format [start_index_1, run_len_1, start_index_2, run_len_2, ...]
        height (int): Height of the original image/mask.
        width (int): Width of the original image/mask.
        
    Returns:
        PIL.Image: Binary mask image (mode 'L').
    """
    if not rle_list:
        return Image.new("L", (width, height), 0)
        
    # Create flat zero array
    mask_flat = np.zeros(height * width, dtype=np.uint8)
    
    # RLE is flat-index based: pairs of (start_index, run_length)
    rle_arr = np.array(rle_list)
    starts = rle_arr[0::2]
    lengths = rle_arr[1::2]
    
    for start, length in zip(starts, lengths):
        mask_flat[start : start + length] = 255
        
    # Reshape back to 2D height x width and convert to PIL Image
    mask_2d = mask_flat.reshape((height, width))
    return Image.fromarray(mask_2d)

# ==========================================
# 2. PyTorch Dataset for VLM Fine-Tuning
# ==========================================
class HEALMedVQADataset(TorchDataset):
    """
    A PyTorch Dataset wrapper for HEAL-MedVQA, designed for fine-tuning
    Vision-Language Models (VLMs) like LLaVA, CogVLM, or PaliGemma.
    """
    def __init__(self, hf_dataset, processor=None, system_prompt=None):
        """
        Args:
            hf_dataset: The Hugging Face split dataset (e.g. ds['train'])
            processor: Hugging Face AutoProcessor (e.g. LlavaProcessor)
            system_prompt (str): Optional system prompt to prepend.
        """
        self.dataset = hf_dataset
        self.processor = processor
        self.system_prompt = system_prompt or "You are a helpful medical assistant."

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # 1. Extract inputs
        image = item["image"]  # PIL Image (L mode - Grayscale)
        question = item["question"]
        answer = item["answer"]
        
        # Ensure image is in RGB for typical pre-trained VLM encoders
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # 2. Construct VLM Template
        # Modify this template depending on your VLM target (e.g., LLaVA, PaliGemma, Qwen-VL)
        # Example LLaVA format:
        prompt = f"<image>\nContext: {self.system_prompt}\nQuestion: {question}\nAnswer: {answer}"
        
        # 3. Process inputs if processor is provided
        if self.processor:
            # Tokenize and encode image
            inputs = self.processor(text=prompt, images=image, padding="max_length", truncation=True, return_tensors="pt")
            # Remove batch dimension
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
            return inputs
            
        # Return raw elements if no processor is specified
        return {
            "image": image,
            "question": question,
            "answer": answer,
            "anatomy": item["anatomy"],
            "question_type": item["question_type"],
            "image_id": item["image_id"]
        }

# ==========================================
# 3. Usage Demonstration
# ==========================================
if __name__ == "__main__":
    print("Loading MM-Hallu/HEAL-MedVQA dataset (train config)...")
    
    # Defensive load of train configuration
    dataset_dict = load_dataset("MM-Hallu/HEAL-MedVQA", "train")
    train_split = dataset_dict["train"]
    
    print(f"Loaded train split size: {len(train_split)} rows")
    
    # Instantiate the PyTorch Dataset
    vqa_dataset = HEALMedVQADataset(train_split)
    
    # Sample the first record
    sample = vqa_dataset[0]
    print("\n--- Processed Sample 0 ---")
    print(f"Question: {sample['question']}")
    print(f"Answer  : {sample['answer']}")
    print(f"Image   : {sample['image']}")
    print(f"Anatomy : {sample['anatomy']}")
    
    # Optional: Decode and inspect the Segmentation Mask (useful if VLM handles bounding boxes/masks)
    first_item = train_split[0]
    if first_item["mask_rle"]:
        print("\nDecoding segmentation mask...")
        mask = decode_rle_mask(
            first_item["mask_rle"], 
            first_item["mask_h"], 
            first_item["mask_w"]
        )
        print(f"Decoded Mask Size: {mask.size} (Height: {first_item['mask_h']}, Width: {first_item['mask_w']})")
        # mask.save("sample_mask.png") # Uncomment to save

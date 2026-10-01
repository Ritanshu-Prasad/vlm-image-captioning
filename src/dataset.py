import os
from datasets import load_dataset
from torch.utils.data import Dataset
from PIL import Image

class COCOCaptionDataset(Dataset):
    """
    A PyTorch Dataset wrapper for the COCO image captioning dataset.
    This will be used to load images and text and pass them to the BLIP processor.
    """
    def __init__(self, split="train", limit=None):
        """
        Args:
            split (str): "train" or "validation"
            limit (int, optional): If set, limits the dataset size (useful for dry-runs).
        """
        print(f"Loading COCO {split} dataset...")
        
        # We use Hugging Face datasets to load COCO directly. 
        # In Kaggle, this will map to their cache automatically if we configure it right.
        self.dataset = load_dataset("HuggingFaceM4/COCO", split=split, trust_remote_code=True)
        
        if limit:
            self.dataset = self.dataset.select(range(limit))
            
        print(f"Loaded {len(self.dataset)} examples.")

    def __len__(self):
        # We mathematically multiply the dataset size by 5.
        # This allows us to use all 5 human annotations per image!
        return len(self.dataset) * 5

    def __getitem__(self, idx):
        # Map the artificially expanded index back to the real image index
        image_idx = idx // 5
        caption_idx = idx % 5
        
        item = self.dataset[image_idx]
        image = item["image"]
        
        raw_caps = item["sentences"]["raw"]
        
        # Fallback if the dataset row surprisingly has less than 5 captions
        if isinstance(raw_caps, list):
            safe_cap_idx = min(caption_idx, len(raw_caps) - 1)
            caption = raw_caps[safe_cap_idx]
        else:
            caption = raw_caps
        
        # --- CONDITIONAL GENERATION LOGIC ---
        word_count = len(caption.split())
        if word_count <= 8:
            prefix = "short caption: "
        else:
            prefix = "detailed caption: "
            
        conditional_caption = prefix + caption
        
        # Enforce RGB to prevent tensor dimension mismatch during batching
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        return {"image": image, "text": conditional_caption}

if __name__ == "__main__":
    print("Validating Dataset Pipeline...")
    test_ds = COCOCaptionDataset(split="validation", limit=5)
    sample = test_ds[0]
    print(f"Validation successful. Output shape: {sample['image'].size}, text: {sample['text']}")

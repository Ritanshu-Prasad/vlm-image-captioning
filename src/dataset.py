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
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        image = item["image"]
        
        raw_caps = item["sentences"]["raw"]
        caption = raw_caps[0] if isinstance(raw_caps, list) else raw_caps
        
        # Enforce RGB to prevent tensor dimension mismatch during batching
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        return {"image": image, "text": caption}

if __name__ == "__main__":
    print("Validating Dataset Pipeline...")
    test_ds = COCOCaptionDataset(split="validation", limit=5)
    sample = test_ds[0]
    print(f"Validation successful. Output shape: {sample['image'].size}, text: {sample['text']}")

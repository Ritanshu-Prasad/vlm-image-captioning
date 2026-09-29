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
        
        # HuggingFace COCO dataset structure usually has 'image' and 'sentences'
        image = item["image"]
        
        # COCO has multiple captions per image. We can just pick the first one for simplicity, 
        # or randomly sample one during training for better robustness.
        # The exact key depends on the dataset version, usually "sentences" -> "raw"
        caption = item["sentences"]["raw"][0] 
        
        # Ensure image is RGB (some might be grayscale)
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        return {"image": image, "text": caption}

if __name__ == "__main__":
    # DRY RUN / DEBUG BLOCK
    # If you run this script directly (python src/dataset.py), it will test the dataloader.
    
    print("Testing Dataset Loader...")
    # Load just 5 examples for testing
    test_ds = COCOCaptionDataset(split="validation", limit=5)
    
    sample = test_ds[0]
    print(f"\nSuccessfully loaded image of size: {sample['image'].size}")
    print(f"Caption: {sample['text']}")
    
    # You can show the image locally to verify
    # sample['image'].show()

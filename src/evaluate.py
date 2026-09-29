"""
evaluate.py

This script evaluates the zero-shot performance of the base BLIP model on the COCO dataset.
It establishes our "Stage 1" baseline metrics before any custom fine-tuning is applied.
"""

import torch
from torch.utils.data import DataLoader
from dataset import COCOCaptionDataset
from model import get_blip_model_and_processor

def collate_fn(batch, processor):
    """
    Prepares images for generation and keeps the raw ground-truth text 
    for metric comparison later.
    """
    images = [item["image"] for item in batch]
    ground_truth_texts = [item["text"] for item in batch]
    
    # We only process the images here because we want the model to GENERATE the text
    inputs = processor(images=images, return_tensors="pt")
    
    return inputs, ground_truth_texts

def main():
    # 1. Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Booting up Zero-Shot Evaluation on device: {device} ---")

    # 2. Load Base Model (NO LoRA for the baseline!)
    # We want to see how the raw model performs out-of-the-box.
    print("Loading Base BLIP Model (Zero-Shot)...")
    processor, model = get_blip_model_and_processor(use_lora=False)
    model.to(device)
    model.eval() # Set model to evaluation mode

    # 3. Load Validation Dataset
    # We use limit=10 for a dry-run to ensure the generation loop works.
    print("Loading validation dataset...")
    val_dataset = COCOCaptionDataset(split="validation", limit=10) 
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=2, 
        shuffle=False, 
        collate_fn=lambda b: collate_fn(b, processor)
    )

    # 4. Evaluation Loop
    print("\nStarting Zero-Shot Generation...")
    
    generated_captions = []
    actual_captions = []
    
    with torch.no_grad(): # Disable gradient calculation to save memory
        for step, (inputs, ground_truth) in enumerate(val_loader):
            
            pixel_values = inputs["pixel_values"].to(device)
            
            # Generate captions! (max_new_tokens limits the length of the sentence)
            outputs = model.generate(pixel_values=pixel_values, max_new_tokens=20)
            
            # Decode the mathematical output back into English words
            decoded_preds = processor.batch_decode(outputs, skip_special_tokens=True)
            
            generated_captions.extend(decoded_preds)
            actual_captions.extend(ground_truth)
            
            # Print the first item in the batch to see it working live
            print(f"\n--- Batch {step} ---")
            print(f"BLIP Guessed : {decoded_preds[0]}")
            print(f"Actual Truth : {ground_truth[0]}")

    print("\nEvaluation Complete! (In the future, we will pass these lists to pycocoevalcap for BLEU scoring).")

if __name__ == "__main__":
    main()

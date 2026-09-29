import torch
from torch.utils.data import DataLoader
import os

from dataset import COCOCaptionDataset
from model import get_blip_model_and_processor

def collate_fn(batch, processor):
    """
    Transforms raw image and text batches into model-ready tensors.
    """
    images = [item["image"] for item in batch]
    texts = [item["text"] for item in batch]
    
    # Process images and pad text for uniform batch dimensions
    inputs = processor(images=images, text=texts, return_tensors="pt", padding=True)
    return inputs

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Booting up Training on device: {device} ---")

    processor, model = get_blip_model_and_processor(use_lora=True)
    model.to(device)

    # Note: limit=20 is currently set for pipeline verification. 
    # Remove limit argument for full training run.
    print("Loading training dataset...")
    train_dataset = COCOCaptionDataset(split="train", limit=20) 
    
    # Batch size tuned for 16GB VRAM environments
    train_loader = DataLoader(
        train_dataset, 
        batch_size=2, 
        shuffle=True, 
        collate_fn=lambda b: collate_fn(b, processor)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    print("\nStarting the training loop...")
    model.train()
    
    epochs = 1
    for epoch in range(epochs):
        for step, batch in enumerate(train_loader):
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            
            # Forward pass: labels=input_ids for autoregressive generation loss
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                pixel_values=pixel_values, 
                labels=input_ids
            )
            
            loss = outputs.loss
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            if step % 2 == 0:
                print(f"Epoch: {epoch} | Step: {step} | Loss: {loss.item():.4f}")
                
    print("\nTraining complete! Saving LoRA adapter weights...")
    output_dir = "./lora_weights"
    model.save_pretrained(output_dir)
    print(f"Weights successfully saved to {output_dir}/adapter_model.safetensors")

if __name__ == "__main__":
    main()

import torch
from torch.utils.data import DataLoader
import os

# Import the dataset and model functions we wrote yesterday
from dataset import COCOCaptionDataset
from model import get_blip_model_and_processor

def collate_fn(batch, processor):
    """
    This function takes a batch of raw images and texts from the dataset
    and converts them into mathematical tensors using the BLIP Processor.
    """
    images = [item["image"] for item in batch]
    texts = [item["text"] for item in batch]
    
    # Process images and pad text so they are all the same length
    inputs = processor(images=images, text=texts, return_tensors="pt", padding=True)
    return inputs

def main():
    # 1. Detect if we have a GPU (CUDA) or if we are on CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Booting up Training on device: {device} ---")

    # 2. Load Model and Processor
    processor, model = get_blip_model_and_processor(use_lora=True)
    model.to(device)

    # 3. Load Dataset 
    # DRYY RUN: We set limit=20 so it only trains on 20 images to verify it works without crashing.
    # To run the full dataset later, just remove `limit=20`.
    print("Loading training dataset...")
    train_dataset = COCOCaptionDataset(split="train", limit=20) 
    
    # 4. Set up DataLoader
    # Batch size of 2 is very safe for 16GB GPUs. 
    train_loader = DataLoader(
        train_dataset, 
        batch_size=2, 
        shuffle=True, 
        collate_fn=lambda b: collate_fn(b, processor)
    )

    # 5. Optimizer (AdamW is standard for Transformers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # 6. The Training Loop
    print("\nStarting the training loop...")
    model.train() # Put model in training mode
    
    epochs = 1
    for epoch in range(epochs):
        for step, batch in enumerate(train_loader):
            
            # Move data to the GPU
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            
            # Forward pass: Feed the images and text to the model.
            # We pass input_ids as the 'labels' so the model can calculate the generation loss.
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                pixel_values=pixel_values, 
                labels=input_ids
            )
            
            loss = outputs.loss
            
            # Backward pass: Calculate gradients
            loss.backward()
            
            # Update the LoRA weights
            optimizer.step()
            optimizer.zero_grad() # Reset gradients for the next step
            
            # Print progress every 2 steps
            if step % 2 == 0:
                print(f"Epoch: {epoch} | Step: {step} | Loss: {loss.item():.4f}")
                
    # 7. Save the LoRA Weights
    print("\nTraining complete! Saving LoRA adapter weights...")
    output_dir = "./lora_weights"
    model.save_pretrained(output_dir)
    print(f"Weights successfully saved to {output_dir}/adapter_model.safetensors")

if __name__ == "__main__":
    main()

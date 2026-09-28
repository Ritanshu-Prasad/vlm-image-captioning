import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from peft import LoraConfig, get_peft_model

def get_blip_model_and_processor(use_lora=True):
    """
    Loads the pre-trained BLIP model and processor from Hugging Face.
    Optionally wraps it in LoRA adapters for memory-efficient fine-tuning on Kaggle GPUs.
    """
    model_id = "Salesforce/blip-image-captioning-base"
    
    print(f"Loading processor: {model_id}")
    processor = BlipProcessor.from_pretrained(model_id)
    
    print(f"Loading base model: {model_id}")
    # We load in standard precision for baseline testing, 
    # but for Kaggle training, we might use torch.float16 or load_in_8bit
    model = BlipForConditionalGeneration.from_pretrained(model_id)
    
    if use_lora:
        print("Injecting LoRA adapters for Parameter-Efficient Fine-Tuning (PEFT)...")
        
        # BLIP has a text decoder and a vision encoder. 
        # We usually target the Self-Attention and Cross-Attention matrices in the text decoder.
        # Finding the exact module names for BLIP: 'query', 'value', 'crossattention' etc.
        # For simplicity, we target 'query' and 'value' in the standard transformer blocks.
        config = LoraConfig(
            r=16, # Rank of the adapter (higher = more capacity, but more memory)
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules=["query", "value"] # Typical attention linear layers
        )
        
        model = get_peft_model(model, config)
        
        # This will print out exactly how many parameters are trainable (usually < 1%)
        model.print_trainable_parameters()
        
    return processor, model

if __name__ == "__main__":
    # DRY RUN / DEBUG BLOCK
    print("Testing Model Loading...")
    
    # Load model WITH LoRA to verify the parameter count
    processor, model = get_blip_model_and_processor(use_lora=True)
    
    print("\nModel successfully loaded and wrapped with LoRA!")

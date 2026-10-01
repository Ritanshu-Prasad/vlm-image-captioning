import torch
from tqdm import tqdm
import evaluate
import json
import os

from model import get_blip_model_and_processor

def load_references_and_generate(model, processor, dataset, device, limit=1000):
    """
    Runs inference on the dataset and collects ground truth references.
    """
    predictions = []
    references = []

    print(f"Generating captions for {limit} validation images...")
    
    for i in tqdm(range(limit)):
        item = dataset[i]
        image = item["image"]
        
        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Ground Truths: COCO has 5 captions per image. We need all of them for fair evaluation.
        raw_caps = item["sentences"]["raw"]
        if isinstance(raw_caps, str):
            refs = [raw_caps]
        else:
            refs = raw_caps # List of 5 strings
            
        references.append(refs)
        
        # Generation
        # For evaluation, we do not prepend "short caption:" unless we are testing that specific prompt.
        # We just want the model's natural description of the image.
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=30)
            
        pred_text = processor.decode(output_ids[0], skip_special_tokens=True).strip()
        predictions.append(pred_text)
        
    return predictions, references

def main():
    # 1. Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Booting up Evaluation on {device} ---")

    processor, model = get_blip_model_and_processor(use_lora=True)
    
    # ⚠️ KAGGLE NOTE: Update this path to wherever your training notebook saved the LoRA weights!
    lora_weights_path = "./lora_weights" 
    if os.path.exists(lora_weights_path):
        print(f"Loading fine-tuned LoRA weights from {lora_weights_path}...")
        model.load_adapter(lora_weights_path)
    else:
        print(f"⚠️ Warning: {lora_weights_path} not found. Evaluating ZERO-SHOT baseline.")

    model.to(device)
    model.eval()

    # 2. Load Dataset
    from datasets import load_dataset
    print("Loading COCO validation dataset...")
    hf_dataset = load_dataset("HuggingFaceM4/COCO", split="validation", trust_remote_code=True)
    
    # Evaluate on a subset to save time (1000 images is standard for robust metrics)
    eval_limit = 1000 
    
    predictions, references = load_references_and_generate(
        model, processor, hf_dataset, device, limit=eval_limit
    )

    # 3. Calculate Metrics
    print("\nCalculating Metrics...")
    
    # BLEU
    bleu_metric = evaluate.load("bleu")
    bleu_results = bleu_metric.compute(predictions=predictions, references=references)
    print(f"BLEU-4: {bleu_results['bleu']:.4f}")

    # ROUGE
    rouge_metric = evaluate.load("rouge")
    rouge_results = rouge_metric.compute(predictions=predictions, references=references)
    print(f"ROUGE-L: {rouge_results['rougeL']:.4f}")

    # METEOR
    meteor_metric = evaluate.load("meteor")
    meteor_results = meteor_metric.compute(predictions=predictions, references=references)
    print(f"METEOR: {meteor_results['meteor']:.4f}")

    # Save results to a file
    results_dict = {
        "BLEU-4": bleu_results['bleu'],
        "ROUGE-L": rouge_results['rougeL'],
        "METEOR": meteor_results['meteor'],
    }
    
    with open("evaluation_metrics.json", "w") as f:
        json.dump(results_dict, f, indent=4)
        
    print("\nMetrics saved to evaluation_metrics.json")
    print("Evaluation Complete!")

if __name__ == "__main__":
    main()

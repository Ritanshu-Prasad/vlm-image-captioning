import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import evaluate
import matplotlib.pyplot as plt
import textwrap

from dataset import COCOCaptionDataset
from transformers import BlipProcessor, BlipForConditionalGeneration
from peft import PeftModel

def load_trained_model(use_lora=True):
    """
    Loads the base BLIP model and optionally attaches our trained LoRA weights.
    """
    model_id = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_id)
    base_model = BlipForConditionalGeneration.from_pretrained(model_id)
    
    if use_lora:
        print("Attaching trained LoRA weights from ./lora_weights...")
        model = PeftModel.from_pretrained(base_model, "./lora_weights")
    else:
        print("Using Zero-Shot Base Model (No LoRA)")
        model = base_model
        
    return processor, model

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Booting up Evaluation on device: {device} ---")

    # Change to False if you want to evaluate the Zero-Shot baseline
    USE_LORA = True 
    processor, model = load_trained_model(use_lora=USE_LORA)
    model.to(device)
    model.eval()

    print("\nLoading validation dataset (Subset of 200 for rapid Kaggle testing)...")
    val_dataset = COCOCaptionDataset(split="validation", limit=200)

    # 1. VISUAL COMPARISON: Did we fix the hallucination?
    print("\n--- Visual Verification ---")
    # Image 0 is the "boy and cow" image where the Zero-Shot model hallucinated "a group of people"
    test_sample = val_dataset[0]
    raw_image = test_sample["image"]
    
    # Test Conditional Generation
    prompt_short = "short caption: "
    prompt_detailed = "detailed caption: "
    
    # Generate Short
    inputs_short = processor(images=raw_image, text=prompt_short, return_tensors="pt").to(device)
    out_short = model.generate(**inputs_short, max_new_tokens=20)
    pred_short = processor.decode(out_short[0], skip_special_tokens=True)
    
    # Generate Detailed
    inputs_detailed = processor(images=raw_image, text=prompt_detailed, return_tensors="pt").to(device)
    out_detailed = model.generate(**inputs_detailed, max_new_tokens=40)
    pred_detailed = processor.decode(out_detailed[0], skip_special_tokens=True)
    
    # Plot the result
    plt.figure(figsize=(8, 8))
    plt.imshow(raw_image)
    plt.axis("off")
    title = f"Conditional LoRA Results:\n\nShort Prompt: '{pred_short}'\nDetailed Prompt: '{pred_detailed}'"
    plt.title(title, fontsize=12, loc='left')
    plt.tight_layout()
    plt.savefig(f"conditional_lora_results.png", dpi=300)
    print("Saved visual proof to 'conditional_lora_results.png'!")

    # 2. QUANTITATIVE METRICS (BLEU, ROUGE, METEOR)
    print("\n--- Quantitative Evaluation ---")
    print("Loading metric calculators (this might download a few small packages)...")
    bleu_calc = evaluate.load("bleu")
    rouge_calc = evaluate.load("rouge")
    meteor_calc = evaluate.load("meteor")
    
    # Import our custom VTAS metric
    from vtas_metric import VTASMetric
    vtas_calc = VTASMetric(device=device)
    
    predictions = []
    references = []
    vtas_scores = []
    
    print("Evaluating model over validation subset...")
    for i in tqdm(range(len(val_dataset))):
        sample = val_dataset[i]
        image = sample["image"]
        ground_truth = sample["text"]
        
        # We don't pass a prompt here to see its natural generalization
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=20)
            
        pred = processor.decode(out[0], skip_special_tokens=True)
        predictions.append(pred)
        
        # For standard metric calculation, references need to be a list of lists
        references.append([ground_truth])
        
        # Calculate VTAS for this specific image and prediction
        vtas_result = vtas_calc.compute(image, pred)
        vtas_scores.append(vtas_result["vtas_score"])
        
    print("\nCalculating Final Scores...")
    bleu_score = bleu_calc.compute(predictions=predictions, references=references)
    rouge_score = rouge_calc.compute(predictions=predictions, references=references)
    meteor_score = meteor_calc.compute(predictions=predictions, references=references)
    avg_vtas = sum(vtas_scores) / len(vtas_scores) if vtas_scores else 0.0
    
    print("\n================ FINAL RESULTS ================")
    print(f"BLEU-4: {bleu_score['bleu']:.4f}")
    print(f"ROUGE-L: {rouge_score['rougeL']:.4f}")
    print(f"METEOR: {meteor_score['meteor']:.4f}")
    print(f"VTAS: {avg_vtas:.4f}")
    print("===============================================")

if __name__ == "__main__":
    main()

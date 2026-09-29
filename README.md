# Experiment: Zero-Shot Baseline Evaluation

## Objective
The purpose of this branch is to establish a performance baseline for the Salesforce BLIP Vision-Language Model on the COCO dataset **before** any Parameter-Efficient Fine-Tuning (PEFT) is applied. 

By measuring the base model's zero-shot inference capabilities, we create a quantifiable benchmark. This ensures that any subsequent LoRA fine-tuning we perform can be scientifically proven to have improved the model's accuracy on our specific domain.

## Methodology
1. **Model:** `Salesforce/blip-image-captioning-base` (Loaded in standard fp32, without LoRA adapters).
2. **Dataset:** COCO 2014 Validation Split.
3. **Task:** Conditional Image Generation (Captioning).
4. **Metrics:** BLEU-4 and CIDEr (via `pycocoevalcap`).

## Execution Instructions
To run this baseline experiment on Kaggle or a local GPU:

```bash
# 1. Ensure dependencies are installed
pip install -r requirements.txt

# 2. Run the evaluation script
python src/evaluate.py
```

## Results & Findings
*(To be populated after the Kaggle run)*
* **BLEU-4 Score:** [TBD]
* **CIDEr Score:** [TBD]

**Hypothesis:** While the pre-trained BLIP model will generate grammatically correct English, it may struggle with the specific formatting or stylistic nuances of the COCO ground-truth captions. We expect our future LoRA fine-tuning to significantly boost the CIDEr score by adapting the model to the COCO specific style.

# Image Captioning with Vision-Language Models (VLMs)

**Computer Vision (CS6350) - TPA-19**

## Problem Statement
This project involves implementing and fine-tuning a vision-language model for image captioning using publicly available datasets (like COCO). The model must learn to generate descriptive captions conditioned on image features using a transformer-based architecture.

## Tech Stack
*   **Frameworks:** PyTorch, Hugging Face `transformers`
*   **Models:** BLIP, ViLT, or similar pre-trained vision-language models
*   **Metrics:** BLEU, CIDEr, METEOR, ROUGE-L
*   **Datasets:** COCO (Common Objects in Context)

## Project Phases
1.  **Environment Setup & Baseline Inference:** Run off-the-shelf VLM for zero-shot captioning.
2.  **Dataset Preparation:** Load and preprocess the COCO dataset using HuggingFace `datasets`.
3.  **Fine-tuning:** Train the model using parameter-efficient methods (PEFT/LoRA).
4.  **Evaluation:** Calculate quantitative metrics and generate qualitative analysis.

## Setup Instructions

1.  Create a virtual environment:
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

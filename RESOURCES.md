# Project Resources: Image Captioning with VLMs

This document contains all the essential research papers, official documentation, and datasets required to build and understand the Vision-Language Model (VLM) for image captioning.

## 1. Core Model Research Papers
These papers define the architectures you are permitted to use for the project. 

*   **BLIP (Highly Recommended):** [Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://arxiv.org/abs/2201.12086) (ICML 2022)
*   **BLIP-2:** [Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597) (ICML 2023)
*   **ViLT:** [Vision-and-Language Transformer Without Convolution or Region Supervision](https://arxiv.org/abs/2102.03334) (ICML 2021)
*   **GIT:** [A Generative Image-to-text Transformer for Vision and Language](https://arxiv.org/abs/2205.14100) (2022)
*   **OFA:** [Unifying Architectures, Tasks, and Modalities Through a Simple Sequence-to-Sequence Learning Framework](https://arxiv.org/abs/2202.03052) (ICML 2022)

## 2. Dataset Papers & Links
*   **COCO (Primary Dataset):** [Microsoft COCO: Common Objects in Context](https://arxiv.org/abs/1405.0312)
    *   *Website/Download:* [cocodataset.org](https://cocodataset.org/)
*   **Flickr30k:** [Flickr30k Entities: Collecting Region-to-Phrase Correspondences](https://arxiv.org/abs/1505.04870)
*   **Conceptual Captions:** [A Cleaned, Hypernymed, Image Alt-text Dataset](https://aclanthology.org/P18-1238/)
*   **(Optional Extension) Fashion Captioning:** [Towards Generating Accurate Descriptions with Semantic Rewards](https://arxiv.org/abs/2008.08336) (ECCV 2020)

## 3. Official Implementation Documentation (Hugging Face)
Since you will be using PyTorch and Hugging Face, these docs are your best friends.

*   **Hugging Face BLIP Documentation:** [BLIP API Reference](https://huggingface.co/docs/transformers/model_doc/blip)
    *   *Includes code snippets for generating captions.*
*   **Hugging Face ViLT Documentation:** [ViLT API Reference](https://huggingface.co/docs/transformers/model_doc/vilt)
*   **Hugging Face Datasets:** [Loading Image Datasets](https://huggingface.co/docs/datasets/image_process)
*   **PEFT (Parameter-Efficient Fine-Tuning):** [LoRA Documentation](https://huggingface.co/docs/peft/index)
    *   *Crucial for fine-tuning large models without running out of GPU memory.*

## 4. Evaluation Metrics
You are required to evaluate your model using BLEU, CIDEr, METEOR, and ROUGE-L. Here are the tools to calculate them:

*   **Hugging Face Evaluate Library:** [evaluate on GitHub](https://github.com/huggingface/evaluate)
*   **pycocoevalcap:** The official COCO caption evaluation code. [GitHub Repository](https://github.com/salaniz/pycocoevalcap)
    *   *This library automatically calculates BLEU, METEOR, ROUGE_L, and CIDEr for you.*

## 5. Tutorials & Guides
*   **Fine-tuning a Vision Transformer:** [Hugging Face Blog](https://huggingface.co/blog/fine-tune-vit) (While this is for classification, the Dataset loading process is identical).
*   **Image Captioning Pipeline Tutorial:** [Transformers Image Captioning Task Guide](https://huggingface.co/tasks/image-captioning)

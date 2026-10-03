# Project TrueSight: Instruction-Tuned VLMs 
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://truesight-vlm.streamlit.app)

## 📖 Overview
Project TrueSight is a comprehensive Machine Learning pipeline designed to combat **"Visual Dominance"** in Vision-Language Models (VLMs). We discovered that standard Foundation Models (like BLIP) often ignore text instructions (e.g., "short caption:" vs "detailed caption:") because their visual encoders overpower their text encoders. 

To solve this, we applied **Instruction-Tuning via LoRA** on a mathematically unrolled dataset (500,000 rows of COCO) and engineered a novel ground-truth hallucination metric called **VTAS (Visual-Truth Alignment Score)** to evaluate the results.

## 🚀 Decoupled Cloud Architecture
This project follows an industry-standard decoupled architecture:
*   **Version Control:** GitHub (`main` branch holds the production-ready code).
*   **Compute (Training):** The model was trained entirely on **Kaggle** utilizing dual **NVIDIA T4 GPUs** (and P100s) to bypass local hardware limitations. Training Iteration 2 processed 500k rows in exactly 8.6 hours.
*   **Deployment (Inference):** The final model is served via a **Streamlit Community Cloud** web application for real-time user interaction. Test the live web app using the badge at the top of this page!

## 🛠️ Tech Stack
*   **Frameworks:** PyTorch, Hugging Face `transformers`, `peft` (LoRA)
*   **Models:** Salesforce BLIP Base, DETR (Object Detection), CLIP (Semantic Fallback)
*   **Metrics:** BLEU, ROUGE-L, METEOR, **VTAS v1.2** (Custom Metric)
*   **Deployment:** Streamlit Cloud

## 🔬 Fine-Tuning Results & Insights

### 1. The Visual Dominance Problem (Iteration 1)
In Iteration 1, we trained the model on 30,000 rows of the COCO dataset. We attempted to perform Instruction Tuning by prompting the model with either `short caption:` or `detailed caption:`. 
However, the model ignored the text prompt entirely and generated the exact same short caption for both prompts. When pushed out-of-distribution, it would even suffer from **Repetition Collapse**, infinitely repeating the word "caption" instead of describing the image.

![Iteration 1 Failure](assets/iter1_results.png)
*Above: Iteration 1 failing to generate a detailed caption.*

### 2. The Solution: Data Unrolling (Iteration 2)
To force the model to learn the text prompt, we mathematically unrolled the PyTorch dataset (`__len__ * 5`), utilizing all 5 human annotations per image without downloading any extra data. This effectively trained the model on 500,000 rows on Kaggle T4 GPUs.
This completely cured the visual dominance! The model learned to flawlessly toggle between short and descriptive sentences purely based on the text prompt, completely overcoming the repetition and ignorance issues from Iteration 1.

![Iteration 2 Success](assets/iter2_results.png)
*Above: Iteration 2 successfully obeying the 'detailed caption' instruction.*

### 3. The Custom VTAS Metric (Quantitative Results)
Standard NLP metrics (BLEU, ROUGE) showed almost **no difference** between the two models because they unfairly penalize descriptive paragraphs that deviate from 5-word ground-truth captions.

To accurately measure the improvement, we engineered **VTAS (Visual-Truth Alignment Score)**. VTAS uses a DETR object detector to find physical objects in the image, spaCy to extract nouns from the generated text, and MiniLM/CLIP to bridge the semantic gap.

**Final Comparison:**
| Metric | Iteration 1 (30k rows) | Iteration 2 (500k rows) |
| :--- | :--- | :--- |
| **BLEU-4** | 0.0598 | 0.0553 |
| **ROUGE-L** | 0.3815 | 0.3763 |
| **METEOR** | 0.3101 | 0.3170 |
| **VTAS (v1.2)** | **0.5692** | **0.6097** 🚀 |

VTAS successfully captured the massive 7% absolute improvement in **Visual Recall**. Iteration 2 successfully describes background objects (trees, dogs, shirts) without hallucinating, which standard metrics completely fail to reward.

### 4. Known Issues & Dataset Bias (Out-of-Distribution Hallucination)
While the model perfectly obeys text instructions, it is strictly bound by the distribution of the **COCO Dataset** (which consists of terrestrial, Earth-based objects). 
During testing, when shown an image of the planet Jupiter, the model accurately generated a descriptive sentence but hallucinated "a thick body of water." It mistook Jupiter's gas clouds for Earth's oceans because it had never seen a gas giant during training. This highlights exactly why objective evaluation metrics like **VTAS** are required to catch domain-specific hallucinations before models reach production.

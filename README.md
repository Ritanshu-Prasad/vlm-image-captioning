# Project TrueSight: Instruction-Tuned VLMs (TPA-19)

**Computer Vision (CS6350) - TPA-19**

## 📖 Overview
Project TrueSight is a comprehensive Machine Learning pipeline designed to combat **"Visual Dominance"** in Vision-Language Models (VLMs). We discovered that standard Foundation Models (like BLIP) often ignore text instructions (e.g., "short caption:" vs "detailed caption:") because their visual encoders overpower their text encoders. 

To solve this, we applied **Instruction-Tuning via LoRA** on a mathematically unrolled dataset (500,000 rows of COCO) and engineered a novel ground-truth hallucination metric called **VTAS (Visual-Truth Alignment Score)** to evaluate the results.

## 🚀 Decoupled Cloud Architecture
This project follows an industry-standard decoupled architecture:
*   **Version Control:** GitHub (`main` branch holds the production-ready code).
*   **Compute (Training):** The model was trained entirely on **Kaggle** utilizing dual **NVIDIA T4 GPUs** (and P100s) to bypass local hardware limitations. Training Iteration 2 processed 500k rows in exactly 8.6 hours.
*   **Deployment (Inference):** The final model is served via a **Streamlit Community Cloud** web application for real-time user interaction.

## 🛠️ Tech Stack
*   **Frameworks:** PyTorch, Hugging Face `transformers`, `peft` (LoRA)
*   **Models:** Salesforce BLIP Base, DETR (Object Detection), CLIP (Semantic Fallback)
*   **Metrics:** BLEU, ROUGE-L, METEOR, **VTAS v1.2** (Custom Metric)
*   **Deployment:** Streamlit, Gradio (Legacy)

## 🔬 Fine-Tuning Results & Insights

### 1. The Visual Dominance Problem (Iteration 1)
In Iteration 1, we trained the model on 30,000 rows of the COCO dataset. We attempted to perform Instruction Tuning by prompting the model with either `short caption:` or `detailed caption:`. 
However, the model ignored the text prompt entirely and generated the exact same short caption for both prompts. 

![Iteration 1 Failure](assets/iter1_results.png)
*Above: Iteration 1 failing to generate a detailed caption.*

### 2. The Solution: Data Unrolling (Iteration 2)
To force the model to learn the text prompt, we mathematically unrolled the PyTorch dataset (`__len__ * 5`), utilizing all 5 human annotations per image without downloading any extra data. This effectively trained the model on 500,000 rows on Kaggle T4 GPUs.

![Iteration 2 Success](assets/iter2_results.png)
*Above: Iteration 2 successfully obeying the 'detailed caption' instruction.*

### 3. The Custom VTAS Metric
Standard NLP metrics (BLEU, ROUGE) showed almost **no difference** between the two models because they unfairly penalize descriptive paragraphs that deviate from 5-word ground-truth captions.

To accurately measure the improvement, we engineered **VTAS (Visual-Truth Alignment Score)**. VTAS uses a DETR object detector to find physical objects in the image, spaCy to extract nouns from the generated text, and MiniLM/CLIP to bridge the semantic gap.

**Final Comparison:**
| Metric | Iteration 1 (30k rows) | Iteration 2 (500k rows) |
| :--- | :--- | :--- |
| **BLEU-4** | 0.0598 | 0.0553 |
| **ROUGE-L** | 0.3815 | 0.3763 |
| **METEOR** | 0.3101 | 0.3170 |
| **VTAS (v1.2)** | **0.5692** | **0.6097** 🚀 |

VTAS successfully captured the massive 7% improvement in **Visual Recall**. Iteration 2 successfully describes background objects (trees, dogs, shirts) without hallucinating, which standard metrics completely fail to reward.

## 💻 Running the Streamlit App Locally

1.  Create a virtual environment:
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Launch the Web UI:
    ```bash
    streamlit run app.py
    ```

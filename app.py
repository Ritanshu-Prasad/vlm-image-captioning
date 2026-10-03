import streamlit as st
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from peft import PeftModel
from PIL import Image

st.set_page_config(page_title="TrueSight VLM", page_icon="👁️", layout="wide")

st.title("👁️ TrueSight Vision-Language Model")
st.markdown("""
### Instruction-Tuned LoRA BLIP Model
Upload an image and see how the **Base BLIP** suffers from 'Visual Dominance', while our **TrueSight LoRA** perfectly obeys the prompt to generate detailed, ground-truth aligned captions!
""")

# Cache the model loading so it doesn't reload on every UI interaction
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    base_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model = PeftModel.from_pretrained(base_model, "lora_weights")
    model.to(device)
    model.eval()
    return processor, model, device

processor, model, device = load_model()

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Uploaded Image', use_column_width=True)
    else:
        image = None

with col2:
    model_choice = st.selectbox(
        "Select Model", 
        ["TrueSight LoRA (Iteration 2)", "Base BLIP (Off-the-shelf)"]
    )
    prompt_choice = st.radio(
        "Select Instruction", 
        ["Detailed Caption", "Short Caption"]
    )
    
    if st.button("Generate Caption", type="primary", use_container_width=True):
        if image is None:
            st.warning("Please upload an image first!")
        else:
            with st.spinner("Generating..."):
                # Construct the prompt
                if prompt_choice == "Short Caption":
                    text_prompt = "short caption :"
                else:
                    text_prompt = "detailed caption :"
                    
                # Toggle LoRA adapters
                if model_choice == "Base BLIP (Off-the-shelf)":
                    model.disable_adapter_layers()
                else:
                    model.enable_adapter_layers()
                    
                # Generate
                inputs = processor(image, text=text_prompt, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=75)
                    
                caption = processor.decode(outputs[0], skip_special_tokens=True)
                caption = caption.replace("short caption : ", "").replace("detailed caption : ", "").strip()
                
                st.success(f"**Generated Caption:** {caption}")
                
st.markdown("---")
st.markdown("""
**How it works:**
* **Base BLIP:** The standard model. It is visually dominant and will likely ignore your text instruction.
* **TrueSight LoRA:** Our fine-tuned model trained on 500,000 mathematically unrolled rows to force Prompt Adherence.
""")

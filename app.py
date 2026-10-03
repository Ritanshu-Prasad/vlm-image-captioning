import streamlit as st
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from peft import PeftModel
from PIL import Image

# Use 'centered' layout instead of 'wide'
st.set_page_config(page_title="TrueSight VLM", page_icon="👁️", layout="centered")

# Center-aligned headers
st.markdown("<h1 style='text-align: center;'>👁️ TrueSight Vision-Language Model</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Instruction-Tuned LoRA BLIP Model</h3>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an image and see how the <b>Base BLIP</b> suffers from 'Visual Dominance', while our <b>TrueSight LoRA</b> perfectly obeys the prompt to generate detailed, ground-truth aligned captions!</p>", unsafe_allow_html=True)
st.markdown("---")

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

# Create a highly focused center column for the entire UI
left_pad, center_col, right_pad = st.columns([1, 3, 1])

with center_col:
    # 1. Image Upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        # Center the image
        st.image(image, use_column_width=True)
    else:
        image = None

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Controls
    model_choice = st.selectbox(
        "Select Model", 
        ["TrueSight LoRA (Iteration 2)", "Base BLIP (Off-the-shelf)"]
    )
    
    # Make radio buttons horizontal to save vertical space
    prompt_choice = st.radio(
        "Select Instruction", 
        ["Detailed Caption", "Short Caption"],
        horizontal=True
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 3. Generate Button
    if st.button("Generate Caption", type="primary", use_container_width=True):
        if image is None:
            st.warning("Please upload an image first!")
        else:
            with st.spinner("Analyzing image..."):
                if prompt_choice == "Short Caption":
                    text_prompt = "short caption :"
                else:
                    text_prompt = "detailed caption :"
                    
                if model_choice == "Base BLIP (Off-the-shelf)":
                    model.disable_adapter_layers()
                else:
                    model.enable_adapter_layers()
                    
                inputs = processor(image, text=text_prompt, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=75)
                    
                caption = processor.decode(outputs[0], skip_special_tokens=True)
                caption = caption.replace("short caption : ", "").replace("detailed caption : ", "").strip()
                
                st.success(f"**Generated Caption:**\n\n{caption}")

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray; font-size: 14px;'><b>How it works:</b><br><b>Base BLIP:</b> The standard model. It is visually dominant and will likely ignore your instruction.<br><b>TrueSight LoRA:</b> Our fine-tuned model trained on 500,000 mathematically unrolled rows to force Prompt Adherence.</p>", unsafe_allow_html=True)

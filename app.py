import gradio as gr
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from peft import PeftModel
from PIL import Image

# Determine the device
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading processor...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

print("Loading base model...")
base_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

print("Loading LoRA adapter...")
# Load the fine-tuned adapter into the base model
model = PeftModel.from_pretrained(base_model, "lora_weights")
model.to(device)
model.eval()

def generate_caption(image, model_choice, prompt_choice):
    """
    Generates a caption based on the selected model and prompt.
    """
    if image is None:
        return "Please upload an image first."
    
    # 1. Construct the text prompt
    if prompt_choice == "Short Caption":
        text_prompt = "short caption :"
    else:
        text_prompt = "detailed caption :"
        
    # 2. Toggle the LoRA adapter based on user choice
    # This is a brilliant PEFT feature: we can turn our fine-tuned weights on/off instantly!
    if model_choice == "Base BLIP (Off-the-shelf)":
        model.disable_adapter_layers()
    else:
        model.enable_adapter_layers()
        
    # 3. Preprocess and Generate
    inputs = processor(image, text=text_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=75)
        
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    
    # Clean up the output string by removing the prompt prefix if the model echoed it
    caption = caption.replace("short caption : ", "").replace("detailed caption : ", "").strip()
    
    return caption

# --- Gradio UI Layout ---
with gr.Blocks(theme=gr.themes.Soft(), title="TrueSight VLM") as demo:
    gr.Markdown(
        """
        # 👁️ TrueSight Vision-Language Model
        ### Instruction-Tuned LoRA BLIP Model
        Upload an image, select a model, and choose an instruction. See how the **Base BLIP** suffers from 'Visual Dominance' (ignoring instructions), while our **TrueSight LoRA** perfectly obeys the prompt to generate detailed, ground-truth aligned captions!
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="pil", label="Upload Image")
            
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=["Base BLIP (Off-the-shelf)", "TrueSight LoRA (Iteration 2)"], 
                    value="TrueSight LoRA (Iteration 2)", 
                    label="Select Model"
                )
                prompt_radio = gr.Radio(
                    choices=["Short Caption", "Detailed Caption"], 
                    value="Detailed Caption", 
                    label="Select Instruction"
                )
                
            submit_btn = gr.Button("Generate Caption", variant="primary")
            
        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="Model Output", 
                lines=5, 
                placeholder="The generated caption will appear here..."
            )
            
            gr.Markdown(
                """
                **How it works:**
                * **Base BLIP:** The standard model. It is visually dominant and will likely ignore whether you ask for a 'short' or 'detailed' caption.
                * **TrueSight LoRA:** Our fine-tuned model trained on 500,000 mathematically unrolled rows to force Prompt Adherence. It will dynamically change its output length based on your instruction.
                """
            )
            
    submit_btn.click(
        fn=generate_caption,
        inputs=[img_input, model_dropdown, prompt_radio],
        outputs=output_text
    )

if __name__ == "__main__":
    print("Launching Gradio App...")
    demo.launch(server_name="0.0.0.0", server_port=7860)

import torch
import spacy
from transformers import DetrImageProcessor, DetrForObjectDetection
from sentence_transformers import SentenceTransformer, util
from PIL import Image

class VTASMetric:
    """
    Visual-Truth Alignment Score (VTAS)
    A custom evaluation metric that grades a generated caption by physically 
    verifying the existence of generated nouns against the raw image pixels using DETR.
    """
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"Loading VTAS modules on {device}...")
        
        # 1. Visual Grounding Module (DETR)
        self.detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(self.device)
        
        # 2. Linguistic Extraction (spaCy)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import os
            print("Downloading spaCy model...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
            
        # 3. Semantic Bridging (all-MiniLM)
        self.semantic_model = SentenceTransformer("all-MiniLM-L6-v2").to(self.device)
        
    def get_visual_objects(self, image: Image):
        """Extracts physical objects from the image using DETR."""
        inputs = self.detr_processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.detr_model(**inputs)
        
        # Filter detections with > 0.9 confidence
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.detr_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
        
        detected_labels = [self.detr_model.config.id2label[label.item()] for label in results["labels"]]
        return list(set(detected_labels)) # Unique physical objects
        
    def get_textual_objects(self, caption: str):
        """Extracts nouns from the generated caption."""
        doc = self.nlp(caption.lower())
        nouns = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
        return list(set(nouns))
        
    def compute(self, image: Image, generated_caption: str):
        """
        Calculates the VTAS score.
        Returns:
            recall: How many visual objects were mentioned.
            hallucination_penalty: How many text objects were hallucinated.
            vtas_score: The final combined score.
        """
        # 1. Extract sets
        V = self.get_visual_objects(image)
        T = self.get_textual_objects(generated_caption)
        
        if not V or not T:
            return {"recall": 0.0, "hallucination_penalty": 0.0, "vtas_score": 0.0, "V": V, "T": T}
            
        # 2. Semantic Intersection
        # Compare every noun in T to every object in V
        v_embeddings = self.semantic_model.encode(V, convert_to_tensor=True)
        t_embeddings = self.semantic_model.encode(T, convert_to_tensor=True)
        
        cosine_scores = util.cos_sim(t_embeddings, v_embeddings)
        
        matches = 0
        for i in range(len(T)):
            # If the text noun has > 0.75 similarity to ANY visual object, it's a match!
            if torch.max(cosine_scores[i]) > 0.75:
                matches += 1
                
        # 3. Math Formulation
        recall = matches / len(V)
        hallucination = (len(T) - matches) / len(T)
        
        # Final Score: Heavily penalize hallucinations
        vtas = max(0.0, recall - (hallucination * 0.5))
        
        return {
            "recall": recall,
            "hallucination_penalty": hallucination,
            "vtas_score": vtas,
            "visual_objects_detected": V,
            "textual_nouns_extracted": T
        }

if __name__ == "__main__":
    import requests
    from io import BytesIO
    print("Testing VTAS Metric Locally...")
    
    # Download a test image (e.g. a cat)
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    
    metric = VTASMetric()
    
    # Fake a bad hallucinated caption
    bad_caption = "A dog sitting on a couch with a frisbee"
    score1 = metric.compute(img, bad_caption)
    print(f"\nBAD CAPTION: {bad_caption}")
    print(f"VTAS: {score1['vtas_score']:.2f} | Detected: {score1['visual_objects_detected']} | Nouns: {score1['textual_nouns_extracted']}")
    
    # Fake a good semantic caption
    good_caption = "Two cats resting on a sofa"
    score2 = metric.compute(img, good_caption)
    print(f"\nGOOD CAPTION: {good_caption}")
    print(f"VTAS: {score2['vtas_score']:.2f} | Detected: {score2['visual_objects_detected']} | Nouns: {score2['textual_nouns_extracted']}")

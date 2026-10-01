import faiss
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

class ImageRetrievalEngine:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cpu"):
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.index = None
        self.metadata = []

    def build_index(self, image_paths):
        self.metadata = image_paths
        embeddings = []
        
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            inputs = self.processor(images=img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                feat = self.model.get_image_features(**inputs)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                embeddings.append(feat.cpu().numpy()[0])
                
        emb_matrix = np.array(embeddings).astype('float32')
        dimension = emb_matrix.shape[1]
        
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product cho cosine similarity
        self.index.add(emb_matrix)

    def search_by_text(self, text_query: str, top_k=5):
        inputs = self.processor(text=[text_query], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_feat = self.model.get_text_features(**inputs)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            
        scores, indices = self.index.search(text_feat.cpu().numpy().astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append({"path": self.metadata[idx], "score": float(score)})
        return results

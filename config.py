import os
import torch

class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
    
    # Device setup
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model Configurations
    CLS_MODEL_NAME = "resnet18"
    YOLO_MODEL_NAME = "yolo11n.pt"
    CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
    EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

    # FAISS paths
    FAISS_IMAGE_INDEX = os.path.join(ARTIFACTS_DIR, "faiss_image.index")
    FAISS_TEXT_INDEX = os.path.join(ARTIFACTS_DIR, "faiss_text.index")

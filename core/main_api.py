from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image
import io
from core.classifier import ImageClassifier
from core.detector import ObjectDetector

app = FastAPI(title="ShopLite Multi-AI Platform APIs")

# Initialize models
classifier = ImageClassifier()
detector = ObjectDetector()

@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "ShopLite AI Services are running"}

@app.post("/api/classify")
async def classify_endpoint(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    result = classifier.predict(image)
    return result

@app.post("/api/detect")
async def detect_endpoint(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    detections, _ = detector.detect(image)
    return {"detections": detections}

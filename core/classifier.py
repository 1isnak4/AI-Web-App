import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class ImageClassifier:
    def __init__(self, num_classes=5, model_path=None, device="cpu"):
        self.device = device
        self.classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
        
        # Load ResNet-18
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image: Image.Image, threshold=0.5):
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
        
        confidence = conf.item()
        label = self.classes[pred_idx.item()]
        
        warning = None
        if confidence < threshold:
            warning = "Độ tin cậy thấp, kết quả dự đoán có thể không chính xác."
            
        return {
            "label": label,
            "confidence": round(confidence, 4),
            "warning": warning
        }

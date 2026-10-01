from ultralytics import YOLO
from PIL import Image
import numpy as np

class ObjectDetector:
    def __init__(self, model_name="yolo11n.pt"):
        self.model = YOLO(model_name)

    def detect(self, image: Image.Image, conf_threshold=0.25):
        results = self.model(image, conf=conf_threshold)[0]
        
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            label = self.model.names[cls_id]
            
            detections.append({
                "label": label,
                "confidence": round(conf, 4),
                "box": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
            })
            
        # Trả về kết quả và ảnh đã vẽ bounding box
        annotated_img_arr = results.plot()
        annotated_img = Image.fromarray(annotated_img_arr[..., ::-1])  # BGR to RGB
        
        return detections, annotated_img

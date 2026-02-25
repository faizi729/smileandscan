from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from pathlib import Path
import io

from PIL import Image
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

app = FastAPI(title="YOLO Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent

basic_yolo_model = None
cancer_yolo_model = None


def load_models():
    global basic_yolo_model, cancer_yolo_model

    if YOLO is None:
        print("⚠️ ultralytics package is not installed. Install it with: pip install ultralytics")
        return

    try:
        basic_weights = BASE_DIR / "basic-detection-yolo" / "best.pt"
        if basic_weights.is_file():
            basic_yolo_model = YOLO(str(basic_weights))
            print(f"✅ Loaded basic YOLO model from {basic_weights}")
        else:
            print(f"⚠️ Basic detection weights not found at {basic_weights}")
    except Exception as e:
        print(f"❌ Failed to load basic YOLO model: {e}")

    try:
        cancer_weights = BASE_DIR / "cancer-detection-yolo" / "best.pt"
        if cancer_weights.is_file():
            cancer_yolo_model = YOLO(str(cancer_weights))
            print(f"✅ Loaded cancer YOLO model from {cancer_weights}")
        else:
            print(f"⚠️ Cancer detection weights not found at {cancer_weights}")
    except Exception as e:
        print(f"❌ Failed to load cancer YOLO model: {e}")


load_models()


class DetectionBox(BaseModel):
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


class BasicDetectionResponse(BaseModel):
    has_detection: bool
    summary: str
    percentage: float
    boxes: List[DetectionBox]


class CancerPredictionResponse(BaseModel):
    prediction: str
    confidence: float
    summary: Optional[str] = None
    boxes: Optional[List[DetectionBox]] = None


def _image_from_upload(file: UploadFile) -> np.ndarray:
    contents = file.file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    return np.array(image)


@app.post("/predict-basic", response_model=BasicDetectionResponse)
async def predict_basic(file: UploadFile = File(...)):
    if basic_yolo_model is None:
        return BasicDetectionResponse(has_detection=False, boxes=[])

    img = _image_from_upload(file)

    # Use tracker for smoother boxes and persistence
    # persist=True tells YOLO this is a sequence of frames
    results = basic_yolo_model.track(source=img, persist=True, conf=0.35, verbose=False)
    r = results[0]

    boxes: List[DetectionBox] = []
    if r.boxes is not None and len(r.boxes) > 0:
        names = r.names
        for box in r.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            boxes.append(
                DetectionBox(
                    class_name=str(names.get(cls_id, cls_id)),
                    confidence=conf,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )

    # Calculate summary on backend
    detection_count = len(boxes)
    if detection_count == 0:
        summary = "No Detection"
        percentage = 0.0
    else:
        # Check for specific classes like calculus / discoloration
        calculus_count = sum(1 for b in boxes if 'calculus' in b.class_name.lower())
        discoloration_count = sum(1 for b in boxes if 'discoloration' in b.class_name.lower() or 'discol' in b.class_name.lower())
        total_issues = calculus_count + discoloration_count

        if total_issues > 5:
            summary = "Heavy Detection"
        elif total_issues > 2:
            summary = "Moderate Detection"
        elif total_issues > 0:
            summary = "Mild Detection"
        else:
            summary = "Detection Found"
        
        # Simple percentage calculation based on detection count
        percentage = min(100.0, (total_issues / 10.0) * 100.0 if total_issues > 0 else (detection_count / 10.0) * 100.0)

    return BasicDetectionResponse(
        has_detection=len(boxes) > 0, 
        summary=summary,
        percentage=float(percentage),
        boxes=boxes
    )


@app.post("/predict-cancer", response_model=CancerPredictionResponse)
async def predict_cancer_yolo(file: UploadFile = File(...)):
    if cancer_yolo_model is None:
        return CancerPredictionResponse(prediction="Unknown", confidence=0.0, boxes=[])

    img = _image_from_upload(file)

    # Use tracker for smoother boxes and persistence
    results = cancer_yolo_model.track(source=img, persist=True, conf=0.35, verbose=False)
    r = results[0]

    boxes: List[DetectionBox] = []
    max_conf = 0.0

    if r.boxes is not None and len(r.boxes) > 0:
        names = r.names
        for box in r.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            boxes.append(
                DetectionBox(
                    class_name=str(names.get(cls_id, cls_id)),
                    confidence=conf,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )
            if conf > max_conf:
                max_conf = conf

    if len(boxes) > 0:
        prediction = "Cancer"
        confidence = max_conf
        summary = "High Risk" if max_conf > 0.7 else "Monitor Required"
    else:
        prediction = "Non-Cancer"
        confidence = 1.0
        summary = "Healthy"

    return CancerPredictionResponse(
        prediction=prediction, 
        confidence=confidence, 
        summary=summary,
        boxes=boxes or None
    )


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "yolo-detection-api",
        "basic_model_loaded": basic_yolo_model is not None,
        "cancer_model_loaded": cancer_yolo_model is not None,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


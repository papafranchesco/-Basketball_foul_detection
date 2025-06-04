import cv2
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from tempfile import NamedTemporaryFile
from rfdetr import RFDETRBase
import torch

WEIGHTS_PATH = "../weights/rf-detr/rf_detr.pt"
SCORE_TH = 0.7

# Load model once at startup
model = RFDETRBase(num_classes=2)
ckpt = torch.load(WEIGHTS_PATH, map_location="cpu")
model.load_state_dict(ckpt)
model.eval()

def process_video(in_path: str, out_path: str):
    cap = cv2.VideoCapture(in_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = model.predict(frame_rgb, conf_thres=SCORE_TH)
        for xyxy, conf, cid in zip(detections.xyxy, detections.confidence, detections.class_id):
            if cid != 1:
                continue
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame_bgr, f"foul {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        writer.write(frame_bgr)
    cap.release()
    writer.release()

app = FastAPI()

@app.post("/annotate")
async def annotate(video: UploadFile = File(...)):
    with NamedTemporaryFile(suffix=".mp4", delete=False) as temp_in:
        temp_in.write(await video.read())
        in_path = temp_in.name
    out_path = in_path.replace(".mp4", "_out.mp4")
    process_video(in_path, out_path)
    return FileResponse(out_path, media_type="video/mp4", filename="annotated.mp4")

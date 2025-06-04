# Web Demo using Colab GPU

This folder contains a minimal FastAPI app to run RF-DETR inference on uploaded videos. It loads the weights from `../weights/rf-detr/rf_detr.pt` and returns an annotated video with detected fouls.

## Running in Colab

1. Clone the repository in a Colab notebook and change into the project directory.
2. Install dependencies:
   ```bash
   pip install -r web_demo/requirements.txt
   ```
3. Launch the API (optionally expose via `ngrok`):
   ```bash
   uvicorn web_demo.app:app --host 0.0.0.0 --port 8000
   ```
4. Send a POST request to `/annotate` with a video file. The endpoint returns the processed video.

You can combine this with `ngrok` or a similar service to provide a public URL for your demo.

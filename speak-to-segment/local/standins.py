"""
CPU stand-ins for the two model endpoints, for running the demo on a laptop.

  /asr/predict  real Whisper (base.en), same contract as the ASR server.
  /sam/predict  NOT SAM 3. CLIPSeg, a small text-prompted segmenter, wrapped to
                return the same response shape as sam/server.py.

    python local/standins.py
    python app.py asr.target_address=http://localhost:11218/asr/predict \
                  sam.target_address=http://localhost:11218/sam/predict
"""

import base64
import cv2
from flask import Flask, jsonify, request
import io
import numpy as np
from pathlib import Path
from PIL import Image
import string
from tempfile import NamedTemporaryFile as TempFile
import torch
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
import whisper

PORT = 11218
CLIPSEG = "CIDAS/clipseg-rd64-refined"
MASK_THRESHOLD = 0.4  # on CLIPSeg's per-pixel probability.
MIN_AREA_FRACTION = 0.002  # drop specks smaller than this share of the image.

app = Flask(__name__)

torch.set_num_threads(4)
asr_model = whisper.load_model("base.en", device="cpu")
seg_processor = CLIPSegProcessor.from_pretrained(CLIPSEG)
seg_model = CLIPSegForImageSegmentation.from_pretrained(CLIPSEG).eval()


@app.route("/health")
def health():
    return jsonify({"status": "healthy"})


@app.route("/asr/predict", methods=["POST"])
def asr():
    audio_bytes = base64.b64decode(request.json["audio"])
    with TempFile(delete=False, suffix=".audio", mode="wb") as tf:
        tf.write(audio_bytes)
    try:
        # whisper shells out to ffmpeg, which sniffs the container itself.
        result = asr_model.transcribe(tf.name, language="en", fp16=False)
    finally:
        Path(tf.name).unlink(missing_ok=True)
    return jsonify({"text": result.get("text", "").strip()})


@app.route("/sam/predict", methods=["POST"])
def sam():
    data = request.json
    image = Image.open(io.BytesIO(base64.b64decode(data["image"]))).convert("RGB")
    # whisper returns "The cat." -- CLIPSeg wants "the cat".
    prompt = data.get("text_prompt", "").strip().strip(string.punctuation).lower()
    return_mode = data.get("return_mode", "contours")

    inputs = seg_processor(text=[prompt], images=[image], return_tensors="pt")
    with torch.no_grad():
        logits = seg_model(**inputs).logits
    prob = torch.sigmoid(logits).squeeze().numpy()
    prob = cv2.resize(prob, image.size, interpolation=cv2.INTER_LINEAR)

    # CLIPSeg gives one heatmap, not instances: split it into connected blobs.
    binary = (prob > MASK_THRESHOLD).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    min_area = MIN_AREA_FRACTION * image.width * image.height

    scores, boxes, contours = [], [], []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        blob = (labels == i).astype(np.uint8)
        found, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not found:
            continue
        largest = max(found, key=cv2.contourArea)
        contours.append(largest.reshape(-1).tolist())
        scores.append(float(prob[labels == i].mean()))
        boxes.append([int(x), int(y), int(x + w), int(y + h)])

    return jsonify(
        {
            "predictions": {
                "masks": [],
                "scores": scores,
                "boxes": boxes,
                "contours": contours if return_mode == "contours" else [],
            },
            "arguments": {"return_mode": return_mode, "model": CLIPSEG},
            "image": {"input": (image.height, image.width)},
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=PORT, debug=False)

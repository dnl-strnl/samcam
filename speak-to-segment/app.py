import base64
from datetime import datetime
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_file,
    send_from_directory,
)
import hydra
import io
import json
from omegaconf import DictConfig
import os
from pathlib import Path
from PIL import Image
import requests
import time
import traceback
import uuid


app = Flask(__name__)

# set by Hydra.
config = None

# segmentation history.
segmentation_history = []

# data storage directories.
AUDIO_DIR = None
IMAGE_DIR = None
MASK_DIR = None
STATIC_DIR = None


def init_app(cfg: DictConfig):
    """Initialize app with Hydra config."""
    global config, AUDIO_DIR, IMAGE_DIR, MASK_DIR, STATIC_DIR
    config = cfg

    # configure data directories.
    AUDIO_DIR = Path(cfg.storage.audio_dir)
    AUDIO_DIR.mkdir(exist_ok=True, parents=True)

    IMAGE_DIR = Path(cfg.storage.image_dir)
    IMAGE_DIR.mkdir(exist_ok=True, parents=True)

    MASK_DIR = Path(cfg.storage.mask_dir)
    MASK_DIR.mkdir(exist_ok=True, parents=True)

    STATIC_DIR = Path(cfg.storage.static_dir)
    STATIC_DIR.mkdir(exist_ok=True, parents=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/static/<path:filename>")
def serve_static(filename):
    """Serve static files."""
    return send_from_directory(STATIC_DIR, filename)


@app.route("/api/config", methods=["GET"])
def get_config():
    """Return endpoint URLs to frontend."""
    return jsonify(
        {
            "asr_endpoint": config.asr.target_address,
            "sam_endpoint": config.sam.target_address,
        }
    )


@app.route("/api/segment", methods=["POST"])
def segment():
    """Full pipeline: record audio -> transcribe text -> obtain masks from SAM

    Streams one JSON object per line, one as each stage finishes ("received",
    "transcribed", then "done" or "error"), so the page can show what it is
    waiting on.
    """
    data = request.json
    audio_base64 = data.get("audio")
    image_base64 = data.get("image")

    if not audio_base64 or not image_base64:
        return jsonify({"error": "No audio or image data"}), 400

    def event(stage, **fields):
        return json.dumps({"stage": stage, **fields}) + "\n"

    def run():
        try:
            # generate IDs for this segmentation.
            seg_id = str(uuid.uuid4())

            # save audio file.
            audio_data = base64.b64decode(audio_base64)
            # the page sends WAV (always-on listening); older clients sent WebM.
            audio_ext = "wav" if audio_data[:4] == b"RIFF" else "webm"
            audio_filename = f"{seg_id}.{audio_ext}"
            audio_path = AUDIO_DIR / audio_filename
            with open(audio_path, "wb") as f:
                f.write(audio_data)

            # save image file.
            image_data = base64.b64decode(image_base64)
            image_filename = f"{seg_id}.png"
            image_path = IMAGE_DIR / image_filename
            with open(image_path, "wb") as f:
                f.write(image_data)

            yield event("received")

            # step 1: send audio to ASR endpoint.
            asr_start = time.time()
            try:
                asr_response = requests.post(
                    config.asr.target_address,
                    json={"audio": audio_base64, "lang": "en"},
                    headers={"Content-Type": "application/json"},
                    timeout=config.asr.timeout,
                )
                asr_response.raise_for_status()
                asr_result = asr_response.json()
                asr_latency = time.time() - asr_start
                text_prompt = asr_result.get("text", "")

                if not text_prompt.strip():
                    yield event("error", error="No text transcribed from audio")
                    return

            except requests.exceptions.RequestException as asr_error:
                print(f"{asr_error=}")
                yield event("error", error=f"ASR endpoint failed: {asr_error=}")
                return

            yield event("transcribed", text_prompt=text_prompt, asr_latency=asr_latency)

            # step 2: send image and text to SAM endpoint.
            sam_start = time.time()
            try:
                sam_payload = {
                    "image": image_base64,
                    "text_prompt": text_prompt,
                    "return_mode": "contours",
                    "conf_threshold": config.sam.conf_threshold,
                }

                sam_response = requests.post(
                    config.sam.target_address,
                    json=sam_payload,
                    headers={"Content-Type": "application/json"},
                    timeout=config.sam.timeout,
                )
                sam_response.raise_for_status()
                sam_result = sam_response.json()
                sam_latency = time.time() - sam_start

                predictions = sam_result.get("predictions", {})
                mask_count = len(predictions.get("scores", []))
                masks = predictions.get("masks", [])
                contours = predictions.get("contours", [])

                scores = predictions.get("scores", [])
                boxes = predictions.get("boxes", [])
                image_info = sam_result.get("image", {})
                arguments = sam_result.get("arguments", {})

                # save masks as JSON.
                mask_filename = f"{seg_id}.json"
                mask_path = MASK_DIR / mask_filename
                with open(mask_path, "w") as f:
                    json.dump(sam_result, f, indent=2)

            except requests.exceptions.RequestException as sam_error:
                print(f"{sam_error=}")
                yield event("error", error=f"sam endpoint failed: {sam_error=}")
                return

            # create history entry.
            history_entry = {
                "id": seg_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "text_prompt": text_prompt,
                "mask_count": mask_count,
                "scores": scores,
                "audio_url": f"/audio/{seg_id}",
                "image_url": f"/image/{seg_id}",
                "asr_latency": round(asr_latency, 3),
                "sam_latency": round(sam_latency, 3),
                "total_latency": round(asr_latency + sam_latency, 3),
            }
            segmentation_history.insert(0, history_entry)

            # keep only `max_history` entries.
            if len(segmentation_history) > config.storage.max_history:
                old_entry = segmentation_history.pop()
                _cleanup_segmentation(old_entry["id"])

            response_data = {
                "success": True,
                "text_prompt": text_prompt,
                "contours": contours,
                "masks": masks,  # Keep for debugging, but won't be used
                "mask_count": mask_count,
                "scores": scores,
                "asr_latency": asr_latency,
                "sam_latency": sam_latency,
                "total_latency": asr_latency + sam_latency,
                "history_entry": history_entry,
            }

            yield event("done", **response_data)

        except Exception as segmentation_error:
            print(f"{segmentation_error=}")
            traceback.print_exc()
            yield event("error", error=f"{segmentation_error=}")

    return Response(run(), mimetype="application/x-ndjson")


@app.route("/audio/<seg_id>")
def get_audio(seg_id):
    """Serve audio file for playback."""
    seg_id = seg_id.replace("..", "").replace("/", "")
    for ext in ("wav", "webm"):
        audio_path = AUDIO_DIR / f"{seg_id}.{ext}"
        if audio_path.exists():
            return send_file(audio_path, mimetype=f"audio/{ext}")

    return jsonify({"error": "Audio file not found"}), 404


@app.route("/image/<seg_id>")
def get_image(seg_id):
    """Serve image file."""
    seg_id = seg_id.replace("..", "").replace("/", "")
    image_path = IMAGE_DIR / f"{seg_id}.png"

    if not image_path.exists():
        return jsonify({"error": "Image file not found"}), 404

    return send_file(image_path, mimetype="image/png")


@app.route("/history", methods=["GET"])
def get_history():
    """Get segmentation history."""
    return jsonify({"history": segmentation_history})


@app.route("/clear_history", methods=["POST"])
def clear_history():
    """Clear all segmentation history."""
    for entry in segmentation_history:
        _cleanup_segmentation(entry["id"])

    segmentation_history.clear()
    return jsonify({"success": True})


def _cleanup_segmentation(seg_id: str):
    """Clean up files associated with a segmentation."""

    # remove audio.
    for ext in ("wav", "webm"):
        (AUDIO_DIR / f"{seg_id}.{ext}").unlink(missing_ok=True)

    # remove image.
    image_path = IMAGE_DIR / f"{seg_id}.png"
    if image_path.exists():
        image_path.unlink()

    # remove masks.
    mask_path = MASK_DIR / f"{seg_id}.json"
    if mask_path.exists():
        mask_path.unlink()


@hydra.main(config_path="config", config_name="speak_to_segment")
def main(cfg: DictConfig):
    init_app(cfg)
    app.run(host=cfg.server.host, port=cfg.server.port, debug=cfg.server.debug)


if __name__ == "__main__":
    main()

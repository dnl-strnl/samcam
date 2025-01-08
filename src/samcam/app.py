import cv2
from datetime import datetime, timedelta
from hydra.utils import instantiate
from flask import Flask, Response, jsonify, render_template, request
import numpy as np
from omegaconf import DictConfig, OmegaConf
import os
from pathlib import Path
import sys

from samcam.stream_buffer import StreamBuffer
from samcam.video_overlay import overlay_mask, overlay_prompts

def make_app():
    app = Flask(__name__)
    cfg = OmegaConf.load(Path(__file__).parent / 'config' / 'app.yaml')

    from sam2.build_sam import build_sam2_camera_predictor
    project_root = Path(__file__).parent.parent.parent
    lib = Path(project_root) / 'lib' / 'segment-anything-2-real-time'

    predictor = build_sam2_camera_predictor(
        cfg.model_cfg,
        cfg.sam2_checkpoint,
        device=cfg.device
    )
    global enable_tracking
    enable_tracking = False

    video_stream = instantiate(cfg.VideoStream)

    @app.route('/')
    def index():
        return render_template('index.html', **video_stream.get_resolution())

    @app.route('/video_feed')
    def video_feed():
        def generate_frames():
            global enable_tracking

            while True:
                frame = video_stream.get_frame()
                if frame is None:
                    continue

                # show overlay of visual prompts temporarily, if recent change.
                if video_stream.show_overlay and \
                   datetime.now() < video_stream.overlay_timeout:
                    frame = video_stream.overlay_image

                # apply tracking overlay, if enabled.
                elif enable_tracking:
                    out_obj_ids, out_mask_logits = predictor.track(frame)
                    frame = overlay_mask(frame, out_obj_ids, out_mask_logits)

                ret, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpg\r\n\r\n' + buffer.tobytes() + b'\r\n'
                )

        return Response(
            generate_frames(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )

    @app.route('/reset_tracker', methods=['POST'])
    def reset_tracker():
        global enable_tracking
        enable_tracking = False
        return jsonify(dict(success='success'))

    @app.route('/process_input', methods=['POST'])
    def process_input():

        data = request.json

        points = data.get('points', [])
        bboxes = data.get('boxes', [])
        labels = np.array([1] * len(points), dtype=np.int32)

        prompt = dict(bbox=bboxes, points=points, labels=labels)

        frame, frame_idx = video_stream.get_current_frame()
        if frame is None:
            return jsonify(dict(error='No frame available.'))

        predictor.load_first_frame(frame)
        predictor.add_new_prompt(frame_idx=0, obj_id=1, **prompt)

        prompt_image = overlay_prompts(frame, points=points, boxes=bboxes)

        video_stream.show_overlay = True
        video_stream.overlay_image = prompt_image
        video_stream.overlay_timeout = datetime.now() + timedelta(seconds=2)

        global enable_tracking
        enable_tracking = True

        ret, buffer = cv2.imencode('.jpg', prompt_image)
        return jsonify(dict(overlay=buffer.tobytes().hex(), timeout=500))

    return app

if __name__ == "__main__":
    app = make_app()
    app.run(debug=True)

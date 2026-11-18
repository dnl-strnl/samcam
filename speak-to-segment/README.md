# speak-to-segment
___
Say what you want segmented. The page records a spoken prompt, Whisper turns it into text, and SAM 3 returns a mask for every matching object in the image.

Three processes, each a URL in `config/speak_to_segment.yaml`:

| Process | Default address | Code |
|---|---|---|
| Flask app that serves the page and calls the two models | `http://localhost:19333` | `app.py`, `templates/` |
| Speech recognition: `{"audio": <base64>, "lang": "en"}` in, `{"text": ...}` out | `http://localhost:11216/predict` | not included; see `local/standins.py` |
| SAM 3: image and text prompt in, contours out | `http://localhost:11217/predict` | `sam/` |

Run everything below from this folder.
___
## Installation

Request access to the SAM 3 checkpoints on [Hugging Face](https://huggingface.co/facebook/sam3), then place `sam3.pt` and `bpe_simple_vocab_16e6.txt.gz` (from `sam3/assets/` in the [SAM 3 repository](https://github.com/facebookresearch/sam3)) in `models/sam3/`.

Build and run the SAM 3 endpoint (needs an NVIDIA GPU):
```bash
sh scripts/sam3_build.sh
sh scripts/sam3_run.sh
```
Install and run the app:
```bash
pip install -r requirements.txt
python app.py
```
Navigate to `http://localhost:19333`, drop an image on the page, click `Listen`, and say what you want segmented.
___
## Without a GPU

`local/standins.py` serves both model endpoints on CPU from one process: Whisper (`base.en`) for speech, and CLIPSeg standing in for SAM 3 with the same response shape. It needs `ffmpeg` on the path.
```bash
pip install -r local/requirements.txt
python local/standins.py
python app.py asr.target_address=http://localhost:11218/asr/predict sam.target_address=http://localhost:11218/sam/predict
```
To use the real SAM 3 on CPU instead, run its server from an environment with [SAM 3](https://github.com/facebookresearch/sam3) and `litserve` installed, and give the app longer timeouts, since a prompt takes several seconds:
```bash
python sam/server.py --config-name sam3 host=127.0.0.1 server.accelerator=cpu endpoint.model_dir=$PWD/models/sam3
python app.py asr.target_address=http://localhost:11218/asr/predict asr.timeout=60 sam.timeout=120
```

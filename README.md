# samcam
___
## Installation

Clone the implementation of the camera predictor adapted for real-time tracking:
```bash
git clone https://github.com/Gy920/segment-anything-2-real-time lib/segment-anything-2-real-time
```
Download the [`sam2.1_hiera_tiny.pt`]() model:
```bash
mkdir -p ./lib/sam2/checkpoints && curl -o ./lib/sam2/checkpoints/sam2.1_hiera_tiny.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
```
Install the `samcam` package:
```bash
poetry install
```
Run the interactive video segmentation application:
```bash
poetry run python -m samcam.app
```
___
## Usage
Navigate to `https://127.0.0.1:5000`:

[insert video here]

Click on the video to add point coordinate prompts for a target object.

[insert video here]

Drag on the video to add bounding box prompts for a target object.

[insert video here]

Click the `Submit & Track` button to initiate tracking of the target object in the live video stream.

[insert video here]

Click `Reset Tracker` to prompt a new target object for tracking:

[insert video here]

Click `Clear Prompts` to refine prompts before submitting prompts for tracking.

[insert video here]

Note that currently, only single-object tracking is supported, so prompting should only be expected to match a single target object to track.

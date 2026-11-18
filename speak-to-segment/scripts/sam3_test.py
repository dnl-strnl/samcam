from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import torch

model = build_sam3_image_model(
    bpe_path='/opt/sam3/weights/bpe_simple_vocab_16e6.txt.gz',
    checkpoint_path='/opt/sam3/weights/sam3.pt',
    load_from_HF=False,
)

processor = Sam3Processor(model)

image = Image.open("/opt/sam3/data/b61372de-6e44-4046-9d05-5b140f0f6061.jpg")
inference_state = processor.set_image(image)

output = processor.set_text_prompt(state=inference_state, prompt="blue cube")

masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

print(boxes.shape)
print(masks.shape)
print(scores.shape)

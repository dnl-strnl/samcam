import base64
import cv2
import hydra
from hydra.utils import instantiate
import io
import logging as log
import litserve
import numpy as np
from omegaconf import DictConfig
from pathlib import Path
from PIL import Image
import torch
from typing import Any, Dict

from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import normalize_bbox


log.getLogger("litserve.server").setLevel(log.WARNING)


class SAM3(litserve.LitAPI):
    def __init__(self, model_dir):
        super().__init__()
        self.model_dir = Path(model_dir)

        self.config = None
        self.model = None
        self.device = None
        self.processor = None
        self.logger = None
        self.default_args = None
        self.image_format = None
        self.image_width = None
        self.image_height = None

    def setup(self, device: str) -> None:
        """Setup SAM3 model with text prompt support"""
        self.device = device

        model_name = Path(self.model_dir).name
        self._setup_logging(f"{model_name}.log")

        self.default_args = {"conf_threshold": 0.5, "return_mode": "contours"}
        self.image_format = "RGB"
        self.image_width = None
        self.image_height = None

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        try:
            self.logger.info("Loading SAM from model directory...")

            # find BPE tokenizer file in model directory.
            bpe_path = self.model_dir / "bpe_simple_vocab_16e6.txt.gz"
            if not bpe_path.exists():
                raise FileNotFoundError(f"No BPE tokenizer not found at {bpe_path}")
            self.logger.info(f"Found BPE tokenizer: {bpe_path}")

            # find checkpoint file in model directory.
            checkpoint_pattern = "*.pt"
            checkpoint_path = next(self.model_dir.glob(checkpoint_pattern), None)
            if checkpoint_path is None or not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"No {checkpoint_pattern} checkpoint found in {self.model_dir}"
                )
            self.logger.info(f"Found checkpoint: {checkpoint_path}")

            # build SAM model with local files.
            self.model = build_sam3_image_model(
                bpe_path=str(bpe_path),
                checkpoint_path=str(checkpoint_path),
                device=device,
                eval_mode=True,
                enable_segmentation=True,
                enable_inst_interactivity=False,
                compile=False,
                load_from_HF=False,
            )
            self.model = self.model.to(self.device)

            confidence_threshold = self.default_args.get("conf_threshold", 0.5)
            self.processor = Sam3Processor(
                self.model,
                device=self.device,
                confidence_threshold=confidence_threshold,
            )

            self.logger.info("SAM loaded successfully! SAM says hello 👋")
        except Exception as load_sam_error:
            self.logger.error(f"{load_sam_error=}")
            raise

        # print memory footprint.
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            self.logger.info(f"VRAM: {allocated=:.1f} MB, {reserved=:.1f} MB")

    def _setup_logging(self, log_file):
        """Setup logging to file."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.model_dir / log_file

        self.logger = log.getLogger(self.model_dir.name)

        # avoid duplicate handlers if setup is called multiple times.
        if not self.logger.handlers:
            # create file handler.
            file_handler = log.FileHandler(log_file)
            file_handler.setLevel(log.INFO)
            # create log formatter.
            formatter = log.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)
            # add handler to logger.
            self.logger.addHandler(file_handler)
            self.logger.setLevel(log.INFO)

    def decode_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Decode incoming request with image and prompt data."""
        json_data = request if "body" not in request else request["body"]

        # decode image.
        image = None
        image_base64 = json_data.get("image", None)
        if image_base64 is not None:
            image_bytes = io.BytesIO(base64.b64decode(image_base64))
            image = Image.open(image_bytes).convert(self.image_format)

        # extract text prompt.
        text_prompt = json_data.get("text_prompt", None)

        # extract box prompts (in xywh format).
        boxes = json_data.get("boxes", None)  # list of [x, y, w, h]
        box_labels = json_data.get("box_labels", None)  # list of True/False

        inference_args = {
            "conf_threshold": json_data.get("conf_threshold", 0.5),
            "return_mode": json_data.get("return_mode", "contours"),
        }

        # override with any other default arguments from config.
        for arg, val in self.default_args.items():
            if arg not in inference_args:
                inference_args[arg] = json_data.get(arg, val)

        return dict(
            image=image,
            text_prompt=text_prompt,
            boxes=boxes,
            box_labels=box_labels,
            **inference_args,
        )

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Main prediction pipeline."""
        pil_image = inputs.pop("image")
        text_prompt = inputs.pop("text_prompt", None)
        boxes = inputs.pop("boxes", None)
        box_labels = inputs.pop("box_labels", None)

        # get confidence threshold from inputs.
        conf_threshold = inputs.get("conf_threshold", 0.5)

        # store input image dimensions.
        self.input_height = pil_image.height
        self.input_width = pil_image.width

        self.logger.info(f"input image: {self.input_width}x{self.input_height}")
        self.logger.info(f"confidence threshold: {conf_threshold}")

        # update processor confidence threshold, if needed.
        if hasattr(self.processor, "confidence_threshold"):
            old_threshold = self.processor.confidence_threshold
            self.processor.confidence_threshold = conf_threshold
            if old_threshold != conf_threshold:
                self.logger.info(
                    f"updated processor confidence threshold: {old_threshold} -> {conf_threshold}"
                )

        # run inference.
        # mixed precision only pays off (and is only well supported) on CUDA.
        use_amp = str(self.device).startswith("cuda")
        with torch.autocast("cuda" if use_amp else "cpu", dtype=torch.bfloat16, enabled=use_amp):
            # set the image in the processor.
            inference_state = self.processor.set_image(pil_image)

            # reset any previous prompts.
            self.processor.reset_all_prompts(inference_state)

            # add text prompt, if provided.
            if text_prompt is not None and text_prompt.strip():
                self.logger.info(f"text prompt: '{text_prompt}'")
                inference_state = self.processor.set_text_prompt(
                    state=inference_state, prompt=text_prompt
                )

            # add box prompts, if provided.
            if boxes is not None and len(boxes) > 0:
                self.logger.info(f"using {len(boxes)} box prompt(s)...")

                # convert boxes from xywh to cxcywh and normalize.
                box_tensor = torch.tensor(boxes).view(-1, 4)
                box_cxcywh = box_xywh_to_cxcywh(box_tensor)
                norm_boxes = normalize_bbox(
                    box_cxcywh, self.input_width, self.input_height
                ).tolist()

                # default all boxes to positive if labels not provided.
                if box_labels is None:
                    box_labels = [True] * len(boxes)

                # add each box as a geometric prompt.
                for box, label in zip(norm_boxes, box_labels):
                    inference_state = self.processor.add_geometric_prompt(
                        state=inference_state, box=box, label=label
                    )

        # postprocess results and return.
        result = self._postprocess(inference_state, **inputs)
        return result

    def _postprocess(
        self, inference_state: Dict[str, Any], return_mode: str = "contours", **kwargs
    ) -> Dict[str, Any]:
        """Postprocess SAM inference results."""

        if "masks" in inference_state and "scores" in inference_state:

            masks_tensor = inference_state["masks"]
            scores_tensor = inference_state["scores"]
            boxes_tensor = inference_state.get("boxes", None)

            # convert tensors to CPU / NumPy.
            if isinstance(masks_tensor, torch.Tensor):
                masks_tensor = masks_tensor.cpu()
            if isinstance(scores_tensor, torch.Tensor):
                scores_tensor = scores_tensor.cpu()
            if boxes_tensor is not None and isinstance(boxes_tensor, torch.Tensor):
                boxes_tensor = boxes_tensor.cpu()

            # masks tensor shape: [num_objects, 1, height, width].
            num_objects = masks_tensor.shape[0]

            masks_list = []
            scores_list = []
            boxes_list = []
            contours_list = []
            for i in range(num_objects):

                # extract mask [1, H, W] -> [H, W].
                mask = masks_tensor[i].squeeze().numpy()
                binary_mask = (mask > 0).astype(np.uint8)

                # extract score.
                score = float(scores_tensor[i].item())
                self.logger.info(
                    f"  object {i}: score={score:.3f}, mask shape={binary_mask.shape}, sum={binary_mask.sum()}"
                )
                scores_list.append(score)

                # extract bounding box.
                if boxes_tensor is not None:
                    box = boxes_tensor[i].numpy().tolist()
                    boxes_list.append(box)
                else:
                    boxes_list.append([])

                # process based on return mode.
                if return_mode == "contours":
                    found_contours, _ = cv2.findContours(
                        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )

                    if found_contours and len(found_contours) > 0:
                        largest_contour = max(found_contours, key=cv2.contourArea)
                        flattened_contour = largest_contour.reshape(-1).tolist()
                        contours_list.append(flattened_contour)
                    else:
                        contours_list.append([])
                else:
                    masks_list.append(binary_mask.tolist())

            result = {
                "predictions": {
                    "masks": masks_list if return_mode == "mask" else [],
                    "scores": scores_list,
                    "boxes": boxes_list,
                },
                "arguments": {
                    "return_mode": return_mode,
                },
                "image": {
                    "input": (self.input_height, self.input_width),
                },
            }

            if return_mode == "contours":
                result["predictions"]["contours"] = contours_list

            return result

        else:
            self.logger.warning("No masks/scores were found in inference state")
            return {
                "predictions": {
                    "masks": [],
                    "scores": [],
                    "boxes": [],
                },
                "arguments": {
                    "return_mode": return_mode,
                },
                "image": {
                    "input": (self.input_height, self.input_width),
                },
            }

    def encode_response(self, output: Dict[str, Any]) -> Dict[str, Any]:
        return output


@hydra.main(config_path="config/server", config_name=Path(__file__).parent.name)
def main(cfg: DictConfig):
    log.info(cfg)

    try:
        lit_api = instantiate(cfg.endpoint)
        lit_srv = instantiate(cfg.server, lit_api=lit_api)

        @lit_srv.app.get("/health")
        async def health_check():
            return {"status": "healthy"}

        @lit_srv.app.get("/task")
        async def get_task():
            return {"task": "sam"}

        @lit_srv.app.get("/model_id")
        async def get_model_id():
            return {"model_id": "sam3"}

        lit_srv.run(host=cfg.host, port=cfg.port)
    except Exception as run_exception:
        log.error(f"{run_exception=}")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

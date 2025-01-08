import cv2
import numpy as np

def overlay_prompts(image, points, boxes):
    prompt_image = image.copy()
    for box in boxes:
        x, y, w, h = map(int, box)
        cv2.rectangle(prompt_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    for point in points:
        x, y = map(int, point)
        cv2.circle(prompt_image, (x, y), 5, (255, 0, 0), -1)
    return prompt_image

def overlay_mask(image, object_ids, mask_logits):
    all_mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)

    for i in range(0, len(object_ids)):
        bin_mask = (mask_logits[i] > 0.0).permute(1, 2, 0)
        out_mask = bin_mask.cpu().numpy().astype(np.uint8) * 255
        all_mask = cv2.bitwise_or(all_mask, out_mask)

    all_mask_rgb = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2RGB)

    all_mask_rgb[:, :, 0] = all_mask_rgb[:, :, 0] * 0
    all_mask_rgb[:, :, 2] = all_mask_rgb[:, :, 2] * 0

    image = cv2.addWeighted(image, 1, all_mask_rgb, 0.5, 0)
    return image

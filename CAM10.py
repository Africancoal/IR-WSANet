# yolov10_gradcam.py
"""
Generate Grad‑CAM heatmaps for **all images in a folder** using a YOLOv10 model (Ultralytics).

### 🩹 Patch #2 – universal output handling
* Fixed `KeyError: 0` by robustly parsing **any** output type (Tensor, list, dict, `Results`).
* Updated `YOLOv10BoxScoreTarget` to accept those types too, so Grad‑CAM works no matter which Ultralytics commit you have.
* Retains silent fallback when `verbose` arg is unsupported.

---
Run:
```bash
python yolov10_gradcam.py
```
It walks `./figures` and writes heat‑maps to `./results`.
"""

from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

try:
    from pytorch_grad_cam.base_cam import BaseCAM  # ≥1.5
except ImportError:
    from pytorch_grad_cam.utils.model_targets import BaseCAM  # ≤1.4

# ───────── Configuration ─────────
WEIGHTS_PATH  = Path('./weights/best.pt')
INPUT_FOLDER  = Path('./figures')
OUTPUT_FOLDER = Path('./results')
IMAGE_EXTS    = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# ───────── Utility: unify model outputs ─────────

def _extract_preds(output):
    """Convert Ultralytics YOLO raw output → Nx6 Tensor [xyxy, conf, cls]."""
    # 1️⃣ Pure tensor
    if torch.is_tensor(output):
        return output

    # 2️⃣ List / tuple
    if isinstance(output, (list, tuple)):
        if len(output) == 0:
            return torch.empty((0, 6))
        return _extract_preds(output[0])

    # 3️⃣ Dict (YOLOv10 training head returns {"one2many", "one2one"})
    if isinstance(output, dict):
        # Prefer post‑NMS "one2one" if present, else fallback to "one2many"
        if 'one2one' in output:
            return _extract_preds(output['one2one'])
        if 'one2many' in output:
            return _extract_preds(output['one2many'])
        # Other known keys
        for k in ('det', 'pred', 'output'):
            if k in output:
                return _extract_preds(output[k])
        # Unknown dict shape
        key_sample = list(output.keys())[:5]
        raise KeyError(f"Unsupported dict keys in model output: {key_sample} …")

    # 4️⃣ Results object (Ultralytics high‑level predict)
    if hasattr(output, 'boxes'):
        b = output.boxes  # Boxes object
        return torch.cat([b.xyxy, b.conf, b.cls], dim=1)

    raise TypeError(f"Unsupported model output type: {type(output)}")(f"Unsupported model output type: {type(output)}")


# ───────── Helper classes ─────────
class YOLOv10BoxScoreTarget(BaseCAM):
    def __init__(self, class_id: int, det_index: int):
        super().__init__()
        self.class_id = class_id
        self.det_index = det_index

    def __call__(self, model_output):
        preds = _extract_preds(model_output)
        conf  = preds[self.det_index, 4]
        cls   = preds[self.det_index, 5]
        return conf * (cls == torch.tensor(self.class_id, device=cls.device)).float()


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = int(round(w * r)), int(round(h * r))
    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
    if (w, h) != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


def preprocess(path: Path, img_size=640):
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = letterbox(rgb, new_shape=img_size).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return tensor, rgb / 255.0


def forward_once(model, tensor):
    """Run model forward, ignoring unknown kwargs, and return unified preds Tensor."""
    try:
        output = model(tensor, verbose=False)
    except TypeError:
        output = model(tensor)
    return _extract_preds(output)


def generate_heatmap(model, img_path: Path, device):
    tensor, rgb_img = preprocess(img_path)
    tensor = tensor.to(device)

    preds = forward_once(model, tensor)
    if preds.shape[0] == 0:
        return (rgb_img * 255).astype(np.uint8)

    best = torch.argmax(preds[:, 4]).item()
    class_id = int(preds[best, 5].item())
    targets = [YOLOv10BoxScoreTarget(class_id, best)]

    with GradCAM(model=model, target_layers=[model.model[-2]], use_cuda=device.type == 'cuda') as cam:
        cam_map = cam(tensor, targets=targets)[0]
    return show_cam_on_image(rgb_img, cam_map, use_rgb=True)


# ───────── Main ─────────

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = YOLO(str(WEIGHTS_PATH)).model.to(device).eval()

    for img_path in INPUT_FOLDER.rglob('*'):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        heatmap = generate_heatmap(model, img_path, device)
        out_path = OUTPUT_FOLDER / f"{img_path.stem}_cam.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
        print(f"✅ Saved: {out_path.relative_to(Path.cwd())}")


if __name__ == '__main__':
    main()

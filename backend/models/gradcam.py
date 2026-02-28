"""
Grad-CAM++ (Gradient-weighted Class Activation Mapping Plus Plus)

Production-quality heatmap system for medical image analysis.
Generates sharp, high-resolution attention maps with bounding boxes
and region coordinates for UI explanation.

Features:
  - Grad-CAM++ for better spatial localization than basic Grad-CAM
  - High-resolution output matching original image dimensions
  - Gamma correction + thresholding for clean, sharp heatmaps
  - Contour extraction with bounding boxes around significant regions
  - Multi-view output: overlay / threshold-highlight / bounding-box
  - Multi-label support (heatmap per predicted class)
  - Region coordinate extraction for frontend UI
  - Single forward+backward pass for performance (<3-4s per image)
"""

import base64
import io
import logging
import time

import cv2
import numpy as np

logger = logging.getLogger("MediScan")

try:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ── Constants ────────────────────────────────────────────────────────
GAMMA = 2.0          # Power-2 scaling: sharpens high-activation peaks
THRESHOLD = 0.65     # Noise gate: only keep strong activations
BLEND_ALPHA = 0.7    # Original image weight in overlay (dominant)
BLEND_BETA = 0.3     # Heatmap weight in overlay (subtle)
MIN_CONTOUR_AREA = 200  # Minimum pixel area for a bounding-box region
BBOX_COLOR = (0, 255, 255)  # Cyan bounding boxes for clean look
BBOX_THICKNESS = 2
CENTER_BIAS_RATIO = 1 / 3  # Radius ratio for center focus circle


# ── Utility: find a layer by dotted path ─────────────────────────────
def _find_target_layer(model, target_layer_name: str):
    """Walk model's named modules to find the layer matching a dotted path."""
    for name, module in model.named_modules():
        if name == target_layer_name:
            return module
    # Fallback: try model.features[-1] for DenseNet-style architectures
    if hasattr(model, 'features'):
        return list(model.features.children())[-1]
    return None


# ── Core: compute Grad-CAM++ activation map ──────────────────────────
def _compute_gradcampp(model, input_tensor, target_layer, class_idx=None, device=None):
    """
    Compute Grad-CAM++ activation map for a single image.

    Grad-CAM++ uses second-order gradients to compute pixel-wise importance
    weights, providing better localization than basic Grad-CAM especially
    when multiple instances of a class appear in the image.

    Args:
        model: PyTorch model in eval mode.
        input_tensor: Preprocessed image tensor (1, C, H, W).
        target_layer: The nn.Module to hook into.
        class_idx: Target class index. If None, uses top predicted class.
        device: torch device.

    Returns:
        cam: numpy array (H, W) normalized to [0, 1], or None on failure.
        class_idx: The class index used for backprop.
    """
    activations = []
    gradients = []

    # Register hooks
    def fwd_hook(module, inp, out):
        activations.append(out.detach())

    def bwd_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    fh = target_layer.register_forward_hook(fwd_hook)
    bh = target_layer.register_full_backward_hook(bwd_hook)

    try:
        model.eval()
        input_tensor = input_tensor.to(device)

        # ── Forward pass ────────────────────────────────────────────
        output = model(input_tensor)
        if output.dim() == 1:
            output = output.unsqueeze(0)

        # Pick target class
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # ── Backward pass on target class score ─────────────────────
        model.zero_grad()
        target_score = output[0, class_idx]
        target_score.backward(retain_graph=False)

        if not activations or not gradients:
            logger.warning("[GRADCAM++] No activations/gradients captured")
            return None, class_idx

        act = activations[0]   # (1, C, H, W)
        grad = gradients[0]    # (1, C, H, W)

        # ── Grad-CAM++ weight computation ───────────────────────────
        # α_kc = grad² / (2·grad² + Σ(act · grad³) + ε)
        grad_sq = grad.pow(2)
        grad_cb = grad.pow(3)
        sum_act_grad_cb = (act * grad_cb).sum(dim=(2, 3), keepdim=True)
        alpha = grad_sq / (2.0 * grad_sq + sum_act_grad_cb + 1e-8)

        # w_k = Σ(α · ReLU(dY/dA))  — pixel-wise importance weights
        weights = (alpha * F.relu(grad)).sum(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activations
        cam = (weights * act).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, class_idx

    finally:
        fh.remove()
        bh.remove()


# ── Lung mask: restrict heatmap to lung region ──────────────────────
def _generate_lung_mask(orig_img):
    """
    Generate a soft lung mask from the original image using thresholding.
    Works well for chest X-rays — removes outer/border noise so the
    heatmap stays focused inside the lungs.

    Returns:
        mask: numpy array (H, W) with values in [0, 1].
    """
    gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    _, lung_mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    lung_mask = cv2.GaussianBlur(lung_mask, (15, 15), 0)
    lung_mask = lung_mask.astype(np.float32) / 255.0
    return lung_mask


# ── Center focus: suppress corner/edge noise ─────────────────────────
def _apply_center_bias(heatmap, ratio=CENTER_BIAS_RATIO):
    """
    Apply a circular center bias that suppresses activations near the
    image corners. Medical models typically focus on central lung/organ
    regions, so this reduces false activations at the edges.

    Args:
        heatmap: numpy array (H, W) in [0, 1].
        ratio: radius expressed as fraction of min(H, W).

    Returns:
        heatmap multiplied by the center bias mask.
    """
    h, w = heatmap.shape
    center_bias = np.zeros_like(heatmap)
    cv2.circle(center_bias, (w // 2, h // 2), int(min(w, h) * ratio), 1.0, -1)
    # Smooth edges so the cutoff isn't abrupt
    center_bias = cv2.GaussianBlur(center_bias, (31, 31), 0)
    center_bias = center_bias / (center_bias.max() + 1e-8)
    return heatmap * center_bias


# ── Post-processing: gamma + threshold + lung mask + center bias ─────
def _postprocess_cam(cam, target_h, target_w, orig_img=None,
                     gamma=GAMMA, threshold=THRESHOLD):
    """
    Enhance the raw CAM for visual clarity.

    Steps:
      1. Gamma correction (power-2) — amplifies high-activation peaks
      2. Threshold at 0.65 — removes low-importance noise
      3. Re-normalize to [0, 1]
      4. Resize to match original image dimensions
      5. Apply lung mask — restrict heatmap to lung region
      6. Apply center bias — suppress corner noise
      7. Final re-normalize
    """
    # 1. Gamma correction (power scaling)
    cam = np.power(cam, gamma)

    # 2. Threshold low-activation noise
    cam[cam < threshold] = 0.0

    # 3. Re-normalize after threshold
    cam_max = cam.max()
    if cam_max > 1e-8:
        cam = cam / cam_max

    # 4. High-quality resize to original image size
    cam_resized = cv2.resize(cam, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    # 5. Apply lung mask (if original image is available)
    if orig_img is not None:
        lung_mask = _generate_lung_mask(orig_img)
        cam_resized = cam_resized * lung_mask

    # 6. Apply center bias to suppress corner noise
    cam_resized = _apply_center_bias(cam_resized)

    # 7. Final re-normalize
    cam_max = cam_resized.max()
    if cam_max > 1e-8:
        cam_resized = cam_resized / cam_max

    return cam_resized


# ── Rendering: generate the different view modes ─────────────────────
def _render_overlay(orig_img, cam, alpha=BLEND_ALPHA, beta=BLEND_BETA):
    """Blend heatmap over original image. Returns BGR uint8 image."""
    cam_uint8 = (cam * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(orig_img, alpha, heatmap_colored, beta, 0)
    return blended


def _render_threshold(orig_img, cam, threshold=0.5):
    """Show only high-activation regions with red tint. Returns BGR uint8 image."""
    mask = (cam > threshold).astype(np.float32)
    # Smooth the mask edges
    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    # Create red highlight
    highlight = orig_img.copy()
    red_tint = np.zeros_like(orig_img)
    red_tint[:, :, 2] = 255  # Red channel
    highlight = cv2.addWeighted(highlight, 0.7, red_tint, 0.3, 0)

    # Composite: original where mask=0, highlight where mask>0
    mask_3ch = np.stack([mask] * 3, axis=-1)
    result = (orig_img * (1 - mask_3ch) + highlight * mask_3ch).astype(np.uint8)
    return result


def _extract_regions(cam, orig_h, orig_w, min_area=MIN_CONTOUR_AREA):
    """
    Extract bounding box regions from high-activation areas.

    Returns:
        regions: list of dicts with keys 'x', 'y', 'w', 'h', 'intensity'
                 Coordinates are relative to original image dimensions.
    """
    # Binary mask of high-activation areas
    cam_uint8 = (cam * 255).astype(np.uint8)
    _, binary = cv2.threshold(cam_uint8, int(0.5 * 255), 255, cv2.THRESH_BINARY)

    # Clean up with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)

        # Mean intensity within this bounding box
        roi = cam[y:y+h, x:x+w]
        intensity = float(np.mean(roi)) if roi.size > 0 else 0.0

        regions.append({
            'x': int(x),
            'y': int(y),
            'w': int(w),
            'h': int(h),
            'intensity': round(intensity, 3)
        })

    # Sort by intensity (most important first)
    regions.sort(key=lambda r: r['intensity'], reverse=True)
    return regions


def _render_bbox(orig_img, cam, regions=None):
    """Draw bounding boxes around significant activation regions."""
    result = orig_img.copy()

    if regions is None:
        h, w = orig_img.shape[:2]
        regions = _extract_regions(cam, h, w)

    for i, r in enumerate(regions):
        x, y, w, h = r['x'], r['y'], r['w'], r['h']
        # Color intensity based on importance (green → yellow → red)
        intensity = r.get('intensity', 0.5)
        color = (
            0,
            int(255 * (1 - intensity)),
            int(255 * intensity)
        )  # BGR: low=green, high=red

        cv2.rectangle(result, (x, y), (x + w, y + h), color, BBOX_THICKNESS)

        # Label with region number
        label = f"R{i+1}: {intensity:.0%}"
        font_scale = max(0.4, min(0.6, w / 200))
        cv2.putText(result, label, (x, max(y - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)

    return result


# ── Encoding helper ──────────────────────────────────────────────────
def _encode_image_b64(img_bgr):
    """Encode a BGR numpy image to base64 PNG string."""
    _, buffer = cv2.imencode('.png', img_bgr)
    return base64.b64encode(buffer).decode('utf-8')


# ══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════

def generate_gradcam(image_path: str, model, target_layer_name: str,
                     device=None, preprocess_fn=None, class_idx=None) -> str | None:
    """
    Generate a Grad-CAM++ heatmap overlay for the given image.

    This is the backward-compatible entry point used by _attach_heatmap().
    Returns a single base64-encoded PNG of the blended heatmap overlay.

    Args:
        image_path: Path to the input image.
        model: PyTorch model (must be in eval mode).
        target_layer_name: Dotted path to target conv layer.
        device: torch device. Defaults to CPU.
        preprocess_fn: Optional callable(image_path) -> tensor.
        class_idx: Target class index. None = top predicted class.

    Returns:
        Base64-encoded PNG string of the overlay heatmap, or None on failure.
    """
    result = generate_gradcam_full(image_path, model, target_layer_name,
                                   device=device, preprocess_fn=preprocess_fn,
                                   class_idx=class_idx)
    if result is None:
        return None
    return result.get('overlay')


def generate_gradcam_full(image_path: str, model, target_layer_name: str,
                           device=None, preprocess_fn=None,
                           class_idx=None) -> dict | None:
    """
    Generate full Grad-CAM++ output with multiple views and region data.

    Returns:
        dict with keys:
            'overlay':    base64 PNG — heatmap blended over original
            'threshold':  base64 PNG — high-activation highlight view
            'bbox':       base64 PNG — bounding boxes on activation regions
            'regions':    list of region dicts [{'x','y','w','h','intensity'}, ...]
            'class_idx':  int — the class index used for heatmap
        Or None on failure.
    """
    if not TORCH_AVAILABLE:
        logger.warning("[GRADCAM++] PyTorch not available")
        return None

    t0 = time.time()

    try:
        if device is None:
            device = torch.device('cpu')

        # ── Find target layer ───────────────────────────────────────
        target_layer = _find_target_layer(model, target_layer_name)
        if target_layer is None:
            logger.warning(f"[GRADCAM++] Target layer '{target_layer_name}' not found")
            return None

        # ── Preprocess input ────────────────────────────────────────
        if preprocess_fn is not None:
            input_tensor = preprocess_fn(image_path)
            if not isinstance(input_tensor, torch.Tensor):
                input_tensor = torch.tensor(input_tensor)
            if input_tensor.dim() == 3:
                input_tensor = input_tensor.unsqueeze(0)
        else:
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
            img = Image.open(image_path).convert('RGB')
            input_tensor = transform(img).unsqueeze(0)

        # ── Load original image at full resolution ──────────────────
        orig_img = cv2.imread(image_path)
        if orig_img is None:
            pil_img = Image.open(image_path).convert('RGB')
            orig_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        orig_h, orig_w = orig_img.shape[:2]

        # ── Compute Grad-CAM++ ──────────────────────────────────────
        cam, used_class = _compute_gradcampp(model, input_tensor, target_layer,
                                              class_idx=class_idx, device=device)
        if cam is None:
            return None

        # ── Post-process: gamma, threshold, lung mask, center bias ────
        cam = _postprocess_cam(cam, orig_h, orig_w, orig_img=orig_img)

        # ── Extract regions (bounding boxes) ────────────────────────
        regions = _extract_regions(cam, orig_h, orig_w)

        # ── Render multiple views ───────────────────────────────────
        overlay_img = _render_overlay(orig_img, cam)
        threshold_img = _render_threshold(orig_img, cam)
        bbox_img = _render_bbox(orig_img, cam, regions=regions)

        # ── Encode to base64 ────────────────────────────────────────
        result = {
            'overlay': _encode_image_b64(overlay_img),
            'threshold': _encode_image_b64(threshold_img),
            'bbox': _encode_image_b64(bbox_img),
            'regions': regions,
            'class_idx': used_class,
        }

        elapsed = time.time() - t0
        logger.info(f"[GRADCAM++] Heatmap generated | layer={target_layer_name} "
                     f"| class={used_class} | regions={len(regions)} | {elapsed:.2f}s")

        return result

    except Exception as e:
        logger.error(f"[GRADCAM++] Failed: {e}")
        return None

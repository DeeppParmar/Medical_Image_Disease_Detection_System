"""
Research-Grade Grad-CAM++ Heatmap System
=========================================

Production-quality, clinically realistic heatmap visualization for
chest X-ray / medical image analysis.

Design principles:
  - Smooth gradients (no blocky patches) via Gaussian smoothing
  - Soft thresholding (no hard cutoffs) preserving natural falloff
  - Lung-region focus with intensity-based soft masking
  - Asymmetry-aware: dominant pathological regions stay stronger
  - Three clean visualization modes:
      Mode 1 (Heatmap)  — smooth Grad-CAM++ overlay, research-paper style
      Mode 2 (Focus)    — soft-threshold highlight, dimmed background
      Mode 3 (Regions)  — bounding boxes on meaningful regions only (max 3)
  - Single forward+backward pass, <3-4s total

Compatible with Flask backend. Output keys match frontend expectations:
  overlay / threshold / bbox / regions / class_idx
"""

import base64
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


# ══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════

BLEND_ALPHA = 0.78       # Original image weight (keep X-ray visible)
BLEND_BETA = 0.22        # Heatmap weight (subtle, not overpowering)
SMOOTH_KSIZE = 11        # Gaussian kernel for initial CAM smoothing (reduced!)
SMOOTH_SIGMA = 5         # Gaussian sigma (was 4, increased for smoother merge)
LUNG_MASK_THRESH = 25    # Intensity threshold for lung mask generation
LUNG_MASK_BLUR = 25      # Gaussian blur kernel for soft lung mask edges
SOFT_THRESH_POWER = 3.0  # Power for soft thresholding (higher = sharper)
SOFT_THRESH_KNEE = 0.35  # Knee point: below this, activations fade rapidly
MIN_CONTOUR_AREA = 500   # Minimum pixel area for region bounding boxes
MAX_REGIONS = 3          # Maximum number of bounding-box regions to show (restored to 3)
BBOX_THICKNESS = 2       # Bounding box line thickness
ASYMMETRY_BOOST = 0.15   # Boost factor for dominant-side emphasis
CONTOUR_NOISE_AREA = 400 # Remove activation blobs smaller than this
PERCENTILE_CUTOFF = 65   # Keep only top 35% activations (lowered for target 20-25% coverage)
SPREAD_POWER = 2.0       # Power compression (2.0 = softer center reduction to allow spread)
EDGE_DECAY_SIGMA_RATIO = 1/3  # Gaussian edge decay width ratio
MAX_COVERAGE_RATIO = 0.65     # Reject heatmaps covering >65% of lung area (was 0.50)


# ══════════════════════════════════════════════════════════════════════
# LAYER FINDER
# ══════════════════════════════════════════════════════════════════════

def _find_target_layer(model, target_layer_name: str):
    """Walk model's named modules to find the layer matching a dotted path."""
    for name, module in model.named_modules():
        if name == target_layer_name:
            return module
    # Fallback: model.features[-1] for DenseNet-style architectures
    if hasattr(model, 'features'):
        return list(model.features.children())[-1]
    return None


# ══════════════════════════════════════════════════════════════════════
# CORE: Grad-CAM++ COMPUTATION
# ══════════════════════════════════════════════════════════════════════

def _compute_gradcampp(model, input_tensor, target_layer,
                       class_idx=None, device=None):
    """
    Compute Grad-CAM++ activation map using second-order gradient weighting.

    Returns:
        cam: numpy array (H, W) normalized to [0, 1], or None on failure.
        class_idx: The class index used.
    """
    activations = []
    gradients = []

    def fwd_hook(module, inp, out):
        activations.append(out.detach())

    def bwd_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    fh = target_layer.register_forward_hook(fwd_hook)
    bh = target_layer.register_full_backward_hook(bwd_hook)

    try:
        model.eval()
        input_tensor = input_tensor.to(device)

        # Forward pass
        output = model(input_tensor)
        if output.dim() == 1:
            output = output.unsqueeze(0)

        # Pick target class
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backward pass on target class score
        model.zero_grad()
        target_score = output[0, class_idx]
        target_score.backward(retain_graph=False)

        if not activations or not gradients:
            logger.warning("[GRADCAM++] No activations/gradients captured")
            return None, class_idx

        act = activations[0]   # (1, C, H, W)
        grad = gradients[0]    # (1, C, H, W)

        # Grad-CAM++ weight computation:
        # α_kc = grad² / (2·grad² + Σ(act · grad³) + ε)
        grad_sq = grad.pow(2)
        grad_cb = grad.pow(3)
        sum_act_grad_cb = (act * grad_cb).sum(dim=(2, 3), keepdim=True)
        alpha = grad_sq / (2.0 * grad_sq + sum_act_grad_cb + 1e-8)

        # w_k = Σ(α · ReLU(dY/dA))
        weights = (alpha * F.relu(grad)).sum(dim=(2, 3), keepdim=True)

        # Weighted combination → ReLU → normalize
        cam = (weights * act).sum(dim=1, keepdim=True)
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


# ══════════════════════════════════════════════════════════════════════
# POST-PROCESSING: Research-grade pipeline
# ══════════════════════════════════════════════════════════════════════

def _smooth_cam(cam, ksize=SMOOTH_KSIZE, sigma=SMOOTH_SIGMA):
    """Apply Gaussian smoothing for research-paper-style gradients."""
    return cv2.GaussianBlur(cam.astype(np.float32), (ksize, ksize), sigma)


def _normalize(cam):
    """Normalize to [0, 1] safely."""
    cmin, cmax = cam.min(), cam.max()
    if cmax - cmin > 1e-8:
        return (cam - cmin) / (cmax - cmin)
    return np.zeros_like(cam)


def _soft_threshold(cam, knee=SOFT_THRESH_KNEE, power=SOFT_THRESH_POWER):
    """
    Soft threshold — preserves smooth gradients instead of hard cutoff.

    Uses a sigmoid-like ramp:  output = cam * sigmoid((cam - knee) * steepness)
    This keeps strong activations mostly intact while smoothly attenuating
    weak ones, avoiding the blocky artifacts of hard thresholding.
    """
    # Steepness controls how sharp the transition is around the knee
    steepness = 10.0
    gate = 1.0 / (1.0 + np.exp(-steepness * (cam - knee)))
    return cam * gate


def _generate_lung_mask(orig_img):
    """
    Generate a soft lung-region mask from the original chest X-ray.

    Uses adaptive thresholding + morphological cleanup to isolate the
    lung fields. The mask is heavily blurred to create soft, natural edges
    that blend smoothly with the heatmap (no hard cutoffs at lung borders).

    Returns:
        mask: numpy (H, W), float32 in [0, 1].
    """
    gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

    # Threshold to isolate body from background
    _, body_mask = cv2.threshold(gray, LUNG_MASK_THRESH, 255, cv2.THRESH_BINARY)

    # Morphological cleanup: close small holes, remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_CLOSE, kernel)
    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_OPEN, kernel)

    # Heavy Gaussian blur for soft edges (no abrupt mask boundaries)
    mask = cv2.GaussianBlur(body_mask, (LUNG_MASK_BLUR, LUNG_MASK_BLUR), 0)
    mask = mask.astype(np.float32) / 255.0

    return mask


def _apply_edge_decay(heatmap, sigma_ratio=EDGE_DECAY_SIGMA_RATIO):
    """
    Apply smooth Gaussian edge decay + explicit border penalty.
    """
    h, w = heatmap.shape
    y, x = np.ogrid[:h, :w]
    sigma = w * sigma_ratio
    center_mask = np.exp(
        -((x - w / 2) ** 2 + (y - h / 2) ** 2) / (2 * sigma ** 2)
    ).astype(np.float32)

    # Explicit region constraint: suppress outer 10%
    pad_y, pad_x = max(1, int(h * 0.10)), max(1, int(w * 0.10))
    roi_mask = np.full((h, w), 0.2, dtype=np.float32)  # 80% penalty outside ROI
    roi_mask[pad_y:h-pad_y, pad_x:w-pad_x] = 1.0
    roi_mask = cv2.GaussianBlur(roi_mask, (15, 15), 0)

    # Hard edge penalty: reduce weight within 10px of true boundary
    roi_mask[:10, :] *= 0.1
    roi_mask[-10:, :] *= 0.1
    roi_mask[:, :10] *= 0.1
    roi_mask[:, -10:] *= 0.1

    return heatmap * center_mask * roi_mask


def _remove_small_blobs(cam, min_area=CONTOUR_NOISE_AREA):
    """
    Remove small disconnected activation blobs using contour filtering.

    Converts to binary, finds contours, zeros out regions smaller than
    min_area, then multiplies back with the original smooth CAM to
    preserve gradients in the kept regions.
    """
    cam_uint8 = (cam * 255).astype(np.uint8)
    _, binary = cv2.threshold(cam_uint8, 25, 255, cv2.THRESH_BINARY)

    # Find and filter contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Build a mask of regions to keep
    keep_mask = np.zeros_like(binary)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(keep_mask, [cnt], -1, 255, -1)

    # Smooth the keep_mask so we don't get hard edges
    keep_mask = cv2.GaussianBlur(keep_mask, (15, 15), 0)
    keep_mask = keep_mask.astype(np.float32) / 255.0

    return cam * keep_mask


def _break_symmetry(cam):
    """
    Reduce unrealistic bilateral symmetry in the heatmap.

    Real pathology is usually asymmetric (one lung more affected).
    This function detects which half has stronger activations and
    slightly boosts it while attenuating the weaker half.
    """
    h, w = cam.shape
    mid = w // 2

    left_mean = cam[:, :mid].mean()
    right_mean = cam[:, mid:].mean()

    if abs(left_mean - right_mean) < 0.02:
        # Nearly identical sides — boost the dominant one
        if left_mean >= right_mean:
            cam[:, :mid] *= (1.0 + ASYMMETRY_BOOST)
            cam[:, mid:] *= (1.0 - ASYMMETRY_BOOST * 0.5)
        else:
            cam[:, mid:] *= (1.0 + ASYMMETRY_BOOST)
            cam[:, :mid] *= (1.0 - ASYMMETRY_BOOST * 0.5)

    return cam


def _limit_active_area(cam, percentile=PERCENTILE_CUTOFF):
    """
    Keep only the top activations by percentile cutoff.

    E.g. percentile=75 keeps only the top 25% of activation values,
    removing the diffuse "full lung glow" effect.
    """
    thresh = np.percentile(cam, percentile)
    cam = cam * (cam >= thresh).astype(np.float32)
    return cam


def _isolate_top_regions(cam, max_blobs=MAX_REGIONS):
    """
    Multi-region separation using connected component labeling.

    Uses scipy.ndimage.label for proper region separation (not contours,
    which merge nearby blobs). Keeps only top N regions by total
    activation mass. Preserves gradient detail within kept regions.
    """
    import scipy.ndimage as ndi

    # Label connected components at 0.30 threshold (accounting for power compression)
    labeled, num_features = ndi.label(cam > 0.30)

    if num_features == 0:
        return cam

    # Compute total activation mass per region
    sizes = ndi.sum(cam, labeled, range(1, num_features + 1))

    # Keep top N regions by activation mass
    top_regions = sorted(
        range(1, num_features + 1),
        key=lambda i: sizes[i - 1],
        reverse=True
    )[:max_blobs]

    mask = np.zeros_like(cam, dtype=np.float32)
    for i in top_regions:
        mask[labeled == i] = 1.0

    # Light smooth on mask edges only (9×9, not heavy)
    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    mask_max = mask.max()
    if mask_max > 1e-8:
        mask = mask / mask_max

    return cam * mask


def _compress_spread(cam, power=SPREAD_POWER):
    """
    Power compression to reduce over-spread.

    Raising to power 2.5 crushes low values aggressively while keeping
    strong peaks intact. This transforms diffuse glow into focused spots.
    """
    return np.power(cam, power)


def _postprocess_cam(cam, target_h, target_w, orig_img=None):
    """
    Full research-grade post-processing pipeline.

    Pipeline order:
      1.  Upscale to full resolution (bicubic)
      2.  Gaussian smoothing (remove pixelation)
      3.  Soft threshold (sigmoid gate, preserve gradients)
      4.  Lung mask (restrict to chest region)
      5.  Percentile cutoff (keep top 25% activations only)
      6.  Power compression (crush low spread, power 2.5)
      7.  Multi-region isolation (keep top 3 blobs only)
      8.  Edge decay (Gaussian falloff from center)
      9.  Remove small noise blobs (contour filter)
      10. Break unrealistic symmetry
      11. Final Gaussian smooth + normalize
    """
    # 1. Upscale to original image dimensions
    cam = cv2.resize(cam, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    # 2. Light smoothing (11×11) — remove pixelation WITHOUT merging regions
    cam = _smooth_cam(cam)
    cam = _normalize(cam)

    # 3. Soft threshold — smooth attenuation of weak activations
    cam = _soft_threshold(cam)
    cam = _normalize(cam)

    # 4. Lung mask — restrict heatmap to lung/body region
    if orig_img is not None:
        lung_mask = _generate_lung_mask(orig_img)
        cam = cam * lung_mask
        cam = _normalize(cam)

    # 5. Percentile cutoff — keep only top 18% activations
    cam = _limit_active_area(cam)
    cam = _normalize(cam)

    # 5b. Morphological opening — remove small noisy speckles after cutoff
    cam_uint8 = (cam * 255).astype(np.uint8)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cam_uint8 = cv2.morphologyEx(cam_uint8, cv2.MORPH_OPEN, morph_kernel)
    cam = cam_uint8.astype(np.float32) / 255.0
    cam = _normalize(cam)

    # 6. Power compression (2.5) — reduce center dominance
    cam = _compress_spread(cam)
    cam = _normalize(cam)

    # 7. Connected-component region separation — keep top 2 blobs
    cam = _isolate_top_regions(cam)
    cam = _normalize(cam)

    # 7b. Coverage rejection — if heatmap covers >50% of lung area, it's
    #     too generic to be clinically useful; dampen significantly
    if orig_img is not None:
        active_pixels = float((cam > 0.1).sum())
        total_pixels = float(cam.size)
        coverage = active_pixels / total_pixels if total_pixels > 0 else 0
        if coverage > MAX_COVERAGE_RATIO:
            # Aggressively narrow: raise to higher power to crush spread
            cam = np.power(cam, 3.5)
            cam = _normalize(cam)
            logger.info(f"[GRADCAM++] Coverage rejection triggered: {coverage:.1%} > {MAX_COVERAGE_RATIO:.0%}, dampened")

    # 8. Gaussian edge decay — suppress border activations
    cam = _apply_edge_decay(cam)

    # 9. Remove small disconnected blobs
    cam = _remove_small_blobs(cam)

    # 10. Break unrealistic symmetry
    cam = _break_symmetry(cam)
    cam = _normalize(cam)

    # 11. Light diffusion (9×9) — not heavy, just clean edges
    cam = cv2.GaussianBlur(cam, (9, 9), 0)

    # 12. Research-style spread: 85% sharp + 15% soft halo
    soft_halo = cv2.GaussianBlur(cam, (25, 25), 0)
    cam = cam * 0.85 + soft_halo * 0.15
    cam = _normalize(cam)

    return cam


# ══════════════════════════════════════════════════════════════════════
# RENDERING: Three visualization modes
# ══════════════════════════════════════════════════════════════════════

def _render_heatmap(orig_img, cam, alpha=BLEND_ALPHA, beta=BLEND_BETA):
    """
    Mode 1: HEATMAP — Research-paper style smooth overlay.

    Smooth Grad-CAM++ blended over original X-ray.
    No bounding boxes, no harsh thresholding.
    X-ray anatomy remains clearly visible.
    """
    # Apply JET colormap to the smooth heatmap
    cam_uint8 = (cam * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)

    # Where CAM is near-zero, we don't want blue tinting from JET.
    # Create a soft mixing mask so zero-CAM areas show pure original.
    mix_mask = cam.copy()
    mix_mask = np.clip(mix_mask * 2.0, 0, 1)  # Ramp up faster
    mix_mask = cv2.GaussianBlur(mix_mask, (7, 7), 0)
    mix_3ch = np.stack([mix_mask] * 3, axis=-1)

    # Blend: where CAM > 0 show overlay; where CAM ≈ 0 show original
    overlay = cv2.addWeighted(orig_img, alpha, heatmap_colored, beta, 0)
    result = (orig_img * (1.0 - mix_3ch) + overlay * mix_3ch).astype(np.uint8)

    return result


def _render_focus(orig_img, cam, focus_threshold=0.30):
    """
    Mode 2: FOCUS — Soft-threshold highlight with dimmed background.

    Strong activation regions are highlighted with a warm color tint.
    Background is slightly dimmed (not black) so anatomical context
    remains visible. Smooth transitions, no hard edges.
    """
    # Create soft focus mask using sigmoid-style ramp
    steepness = 8.0
    focus_mask = 1.0 / (1.0 + np.exp(-steepness * (cam - focus_threshold)))
    focus_mask = cv2.GaussianBlur(focus_mask.astype(np.float32), (15, 15), 0)
    focus_3ch = np.stack([focus_mask] * 3, axis=-1)

    # Dimmed background (60% brightness)
    dimmed = (orig_img * 0.4).astype(np.uint8)

    # Warm highlight tint on focused regions
    warm_tint = orig_img.copy().astype(np.float32)
    warm_tint[:, :, 2] = np.minimum(warm_tint[:, :, 2] * 1.3, 255)  # Boost red
    warm_tint[:, :, 1] = warm_tint[:, :, 1] * 0.95  # Slight green reduction
    warm_tint = warm_tint.astype(np.uint8)

    # Composite: dimmed background + warm-tinted focus areas
    result = (dimmed * (1.0 - focus_3ch) + warm_tint * focus_3ch).astype(np.uint8)

    return result


def _extract_regions(cam, orig_h, orig_w, min_area=MIN_CONTOUR_AREA,
                     max_regions=MAX_REGIONS):
    """
    Extract the top meaningful bounding-box regions from the heatmap.

    Uses morphological cleanup + contour detection. Limits output to
    max_regions (default 3) largest/strongest regions to avoid clutter.
    """
    cam_uint8 = (cam * 255).astype(np.uint8)
    otsu_val, _ = cv2.threshold(cam_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    manual_thresh = int(otsu_val * 0.60)  # softer threshold: 60% of OTSU
    _, binary = cv2.threshold(cam_uint8, manual_thresh, 255, cv2.THRESH_BINARY)

    # Morphological cleanup: close gaps, remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)

        # Mean activation intensity within this bounding box
        roi = cam[y:y+h, x:x+w]
        intensity = float(np.mean(roi)) if roi.size > 0 else 0.0

        regions.append({
            'x': int(x), 'y': int(y),
            'w': int(w), 'h': int(h),
            'intensity': round(intensity, 3)
        })

    # Sort by intensity (strongest first), limit to max_regions
    regions.sort(key=lambda r: r['intensity'], reverse=True)
    return regions[:max_regions]


def _render_regions(orig_img, cam, regions=None):
    """
    Mode 3: REGIONS — Clean bounding boxes on meaningful areas only.

    No noise boxes. Maximum 3 regions. Color-coded by intensity:
      High intensity → red, Medium → orange, Low → yellow.
    Labels show region number and intensity percentage.
    """
    result = orig_img.copy()

    if regions is None:
        h, w = orig_img.shape[:2]
        regions = _extract_regions(cam, h, w)

    for i, r in enumerate(regions):
        x, y, w, h = r['x'], r['y'], r['w'], r['h']
        intensity = r.get('intensity', 0.5)

        # Color gradient: yellow → orange → red based on intensity
        # BGR format
        red = int(np.clip(255 * intensity, 0, 255))
        green = int(np.clip(255 * (1.0 - intensity * 0.8), 0, 255))
        blue = 0
        color = (blue, green, red)

        # Draw rounded-feel rectangle (thicker for higher intensity)
        thickness = max(BBOX_THICKNESS, int(intensity * 3) + 1)
        cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)

        # Label: region number + intensity
        label = f"R{i+1}: {intensity:.0%}"
        font_scale = max(0.45, min(0.65, w / 180))

        # Label background for readability
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                      font_scale, 1)
        label_y = max(y - 8, th + 4)
        cv2.rectangle(result, (x, label_y - th - 4),
                      (x + tw + 6, label_y + 2), color, -1)
        cv2.putText(result, label, (x + 3, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), 1, cv2.LINE_AA)

    return result


# ══════════════════════════════════════════════════════════════════════
# ENCODING HELPER
# ══════════════════════════════════════════════════════════════════════

def _encode_image_b64(img_bgr):
    """Encode a BGR numpy image to base64 PNG string."""
    _, buffer = cv2.imencode('.png', img_bgr)
    return base64.b64encode(buffer).decode('utf-8')


# ══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════

def generate_gradcam(image_path: str, model, target_layer_name: str,
                     device=None, preprocess_fn=None, class_idx=None):
    """
    Backward-compatible entry point. Returns base64 heatmap overlay PNG.
    """
    result = generate_gradcam_full(image_path, model, target_layer_name,
                                   device=device, preprocess_fn=preprocess_fn,
                                   class_idx=class_idx)
    if result is None:
        return None
    return result.get('overlay')


def generate_gradcam_full(image_path: str, model, target_layer_name: str,
                           device=None, preprocess_fn=None,
                           class_idx=None):
    """
    Generate research-grade Grad-CAM++ output with three visualization modes.

    Returns:
        dict with keys (matching frontend expectations):
            'overlay':    base64 PNG — Mode 1: smooth heatmap overlay
            'threshold':  base64 PNG — Mode 2: soft-focus highlight
            'bbox':       base64 PNG — Mode 3: region bounding boxes
            'regions':    list of region dicts [{'x','y','w','h','intensity'}]
            'class_idx':  int — class index used for heatmap
        Or None on failure.
    """
    if not TORCH_AVAILABLE:
        logger.warning("[GRADCAM++] PyTorch not available")
        return None

    t0 = time.time()

    try:
        if device is None:
            device = torch.device('cpu')

        # ── Find target layer ────────────────────────────────────────
        target_layer = _find_target_layer(model, target_layer_name)
        if target_layer is None:
            logger.warning(
                f"[GRADCAM++] Target layer '{target_layer_name}' not found")
            return None

        # ── Preprocess input tensor ──────────────────────────────────
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

        # ── Load original image at full resolution ───────────────────
        orig_img = cv2.imread(image_path)
        if orig_img is None:
            pil_img = Image.open(image_path).convert('RGB')
            orig_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        orig_h, orig_w = orig_img.shape[:2]

        # ── Compute Grad-CAM++ ───────────────────────────────────────
        cam, used_class = _compute_gradcampp(
            model, input_tensor, target_layer,
            class_idx=class_idx, device=device)
        if cam is None:
            return None

        # ── Post-process: full research-grade pipeline ───────────────
        cam = _postprocess_cam(cam, orig_h, orig_w, orig_img=orig_img)

        # ── Extract regions (for Mode 3 + frontend data) ─────────────
        regions = _extract_regions(cam, orig_h, orig_w)

        # ── Render three visualization modes ─────────────────────────
        heatmap_img = _render_heatmap(orig_img, cam)       # Mode 1
        focus_img = _render_focus(orig_img, cam)            # Mode 2
        regions_img = _render_regions(orig_img, cam, regions)  # Mode 3

        # ── Encode to base64 (keys match frontend expectations) ──────
        result = {
            'overlay': _encode_image_b64(heatmap_img),
            'threshold': _encode_image_b64(focus_img),
            'bbox': _encode_image_b64(regions_img),
            'regions': regions,
            'class_idx': used_class,
        }

        elapsed = time.time() - t0
        logger.info(
            f"[GRADCAM++] Heatmap generated | layer={target_layer_name} "
            f"| class={used_class} | regions={len(regions)} | {elapsed:.2f}s")

        return result

    except Exception as e:
        logger.error(f"[GRADCAM++] Failed: {e}")
        return None

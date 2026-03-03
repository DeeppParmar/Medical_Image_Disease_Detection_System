"""
Medical Image Validator — MediScan AI
======================================
Validates uploaded images as genuine X-ray or CT scans before
routing to disease detection models.

SCOPE: X-ray (chest, bone) and CT scans only.
       Will reject valid medical images from other modalities
       (fundus, dermatoscopy, histopathology, ultrasound).
       This is intentional — MediScan AI does not support those modalities.

WEIGHTS: Pre-trained binary classifier at datasets/medical_validator_weights.pth
         Training data has been deleted post-training. Do not attempt to retrain.
         The weights are the source of truth. Treat them as read-only.

KNOWN EDGE CASES THAT WILL PASS (acceptable false negatives):
  - Black-and-white film photographs (rare)
  - Grayscale electron microscopy scans (rare)
  - Dental panoramic X-rays (intentionally narrow scope)

KNOWN EDGE CASES THAT WILL REJECT (acceptable false positives to document):
  - Screenshots of X-rays taken on a phone screen (JPEG glare/color added)
  - X-rays with heavy colored annotation overlays
  - Pediatric hand X-rays (unusual contrast profile — test manually)
"""

import os
import time
import logging
import json
from datetime import datetime, timezone

import cv2
import numpy as np

logger = logging.getLogger("MediScan")

# ── Structured JSON log helper ─────────────────────────────────────────
def _log_event(payload: dict):
    """Structured event logger for audit trail."""
    payload["timestamp"] = datetime.now(timezone.utc).isoformat()
    logger.info(json.dumps(payload))


# ══════════════════════════════════════════════════════════════════════
# CNN VALIDATOR (Lazy-loaded)
# ══════════════════════════════════════════════════════════════════════

WEIGHTS_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'datasets', 'medical_validator_weights.pth'
)

# Check for torch availability (matches gradcam.py pattern)
try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class MedicalCNNValidator:
    """
    ResNet18 backbone with fine-tuned binary classification head.
    Backbone weights: ImageNet pretrained (frozen).
    Head weights: Loaded from WEIGHTS_PATH (trained on medical vs non-medical).
    Output: Single float [0.0, 1.0] where 1.0 = medical scan.
    """

    def __init__(self):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")

        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Freeze entire backbone — we only use learned head weights
        for param in backbone.parameters():
            param.requires_grad = False

        # Replace FC layer — must match architecture used during training
        backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.model = backbone
        self.model.eval()

    def predict(self, tensor):
        """Run inference on preprocessed tensor. Returns float [0, 1]."""
        with torch.no_grad():
            return self.model(tensor).item()


# Preprocessing pipeline — must be identical to what was used during training
_cnn_transform = None
if TORCH_AVAILABLE:
    _cnn_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

# Module-level singleton — lazy loaded, never reloaded after first call
_validator = None
_validator_mode = "uninitialized"  # "cnn_weights" | "heuristic_fallback"


def _get_validator():
    """
    Lazy-loads the CNN validator on first call.
    Returns (validator_instance_or_None, mode_string).
    """
    global _validator, _validator_mode

    if _validator_mode != "uninitialized":
        return _validator, _validator_mode

    if not TORCH_AVAILABLE:
        _log_event({
            "event": "validator_init",
            "status": "fallback",
            "reason": "PyTorch not available",
            "mode": "heuristic_fallback"
        })
        _validator = None
        _validator_mode = "heuristic_fallback"
        return None, "heuristic_fallback"

    abs_weights = os.path.abspath(WEIGHTS_PATH)
    if not os.path.exists(abs_weights):
        _log_event({
            "event": "validator_init",
            "status": "fallback",
            "reason": f"Weights not found at {abs_weights}",
            "mode": "heuristic_fallback"
        })
        _validator = None
        _validator_mode = "heuristic_fallback"
        return None, "heuristic_fallback"

    try:
        instance = MedicalCNNValidator()
        state_dict = torch.load(abs_weights, map_location="cpu", weights_only=True)
        instance.model.load_state_dict(state_dict)
        instance.model.eval()
        _validator = instance
        _validator_mode = "cnn_weights"
        _log_event({
            "event": "validator_init",
            "status": "success",
            "weights_path": abs_weights,
            "mode": "cnn_weights"
        })
    except Exception as e:
        _log_event({
            "event": "validator_init",
            "status": "error",
            "error": str(e),
            "mode": "heuristic_fallback"
        })
        _validator = None
        _validator_mode = "heuristic_fallback"

    return _validator, _validator_mode


# ══════════════════════════════════════════════════════════════════════
# HEURISTIC ENGINE
# ══════════════════════════════════════════════════════════════════════

def _compute_heuristic(bgr_img: np.ndarray) -> dict:
    """
    Fast pre-filter using 7 calibrated image statistics.
    Calibrated specifically for chest X-ray and CT scan characteristics.
    Returns score (0.0–1.0) and per-flag breakdown.
    """
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # --- HARD REJECT: Resolution bounds ---
    if h < 64 or w < 64:
        return {"score": 0.0, "passed": False,
                "hard_reject": "resolution_too_small", "flags": {}, "details": {}}
    if h > 8192 or w > 8192:
        return {"score": 0.0, "passed": False,
                "hard_reject": "resolution_too_large", "flags": {}, "details": {}}

    flags = {}
    details = {}

    # FLAG 1: Grayscale dominance
    # X-rays exported as JPEG/PNG may have slight color noise (threshold: 18)
    # Selfies, memes, color photos score 60–150+
    b, g, r = cv2.split(bgr_img)
    saturation = (np.maximum(np.maximum(r.astype(int), g.astype(int)),
                             b.astype(int)) -
                  np.minimum(np.minimum(r.astype(int), g.astype(int)),
                             b.astype(int)))
    mean_sat = float(saturation.mean())
    flags["grayscale_dominant"] = mean_sat < 18
    details["mean_saturation"] = round(mean_sat, 2)

    # FLAG 2: Intensity dynamic range
    p5 = float(np.percentile(gray, 5))
    p95 = float(np.percentile(gray, 95))
    dynamic_range = p95 - p5
    flags["adequate_dynamic_range"] = dynamic_range > 80
    details["dynamic_range"] = round(dynamic_range, 2)

    # FLAG 3: Contrast (standard deviation of pixel values)
    std_dev = float(gray.std())
    flags["valid_contrast"] = 25.0 < std_dev < 120.0
    details["pixel_std_dev"] = round(std_dev, 2)

    # FLAG 4 + 5: Exposure checks
    bright_ratio = float((gray > 240).mean())
    dark_ratio = float((gray < 15).mean())
    flags["not_overexposed"] = bright_ratio < 0.35
    flags["not_underexposed"] = dark_ratio < 0.55  # Relaxed for CT (large dark areas)
    details["bright_pixel_ratio"] = round(bright_ratio, 3)
    details["dark_pixel_ratio"] = round(dark_ratio, 3)

    # HARD REJECT: completely blank images
    if bright_ratio > 0.85:
        return {"score": 0.0, "passed": False,
                "hard_reject": "solid_white", "flags": flags, "details": details}
    if dark_ratio > 0.85:
        return {"score": 0.0, "passed": False,
                "hard_reject": "solid_black", "flags": flags, "details": details}

    # FLAG 6: Edge structure (Canny density)
    # Medical scans: 0.02–0.30. Blank/uniform: near 0. Chaotic photos: >0.35
    edges = cv2.Canny(gray, 30, 100)
    edge_density = float((edges > 0).mean())
    flags["has_edge_structure"] = 0.02 < edge_density < 0.35
    details["edge_density"] = round(edge_density, 4)

    # FLAG 7: LBP texture variance (strongest anti-selfie/anti-natural-scene)
    # Natural scenes: LBP variance 3000–10000+
    # X-rays/CT: LBP variance 200–1800 (smooth gradients, structured edges)
    lbp_map = np.zeros_like(gray, dtype=np.uint8)
    # Downsample for speed on large images
    if h > 512 or w > 512:
        scale = 512.0 / max(h, w)
        small = cv2.resize(gray, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
    else:
        small = gray

    pad = np.pad(small, 1, mode='edge')
    center = pad[1:-1, 1:-1].astype(int)
    neighbors = [
        pad[0:-2, 0:-2], pad[0:-2, 1:-1], pad[0:-2, 2:],
        pad[1:-1, 2:],   pad[2:,   2:],   pad[2:,   1:-1],
        pad[2:,   0:-2], pad[1:-1, 0:-2]
    ]
    lbp_map = np.zeros_like(small, dtype=np.uint8)
    for i, nb in enumerate(neighbors):
        lbp_map += ((nb.astype(int) >= center).astype(np.uint8) << i)
    lbp_variance = float(lbp_map.var())
    flags["not_natural_scene"] = lbp_variance < 2800
    details["lbp_variance"] = round(lbp_variance, 1)

    # --- HARD REJECT COMBOS ---
    # Both colorful AND complex texture = definitely not X-ray
    if not flags["grayscale_dominant"] and not flags["not_natural_scene"]:
        return {
            "score": 0.0, "passed": False,
            "hard_reject": "colorful_natural_scene",
            "flags": flags, "details": details
        }

    # --- COMPOSITE SCORE ---
    weights = {
        "grayscale_dominant":       0.30,
        "adequate_dynamic_range":   0.15,
        "valid_contrast":           0.15,
        "not_overexposed":          0.08,
        "not_underexposed":         0.08,
        "has_edge_structure":       0.12,
        "not_natural_scene":        0.12,
    }
    score = sum(w for k, w in weights.items() if flags.get(k, False))

    # Decision zones
    if score >= 0.60:
        passed = True
    elif score >= 0.38:
        passed = None   # Ambiguous — defer to CNN
    else:
        passed = False

    return {
        "score": round(score, 3),
        "passed": passed,
        "hard_reject": None,
        "flags": flags,
        "details": details
    }


# ══════════════════════════════════════════════════════════════════════
# REJECTION MESSAGE BUILDER
# ══════════════════════════════════════════════════════════════════════

def _rejection_message(flags: dict, hard_reject=None) -> str:
    """Returns a specific, actionable rejection reason for the UI."""
    if hard_reject == "resolution_too_small":
        return ("Image resolution is too low for analysis. "
                "Please upload a high-resolution medical scan.")
    if hard_reject == "resolution_too_large":
        return ("Image resolution exceeds supported limits. "
                "Please upload a standard medical scan image.")
    if hard_reject == "solid_white":
        return ("Image appears to be blank (all white). "
                "Please upload a valid medical scan.")
    if hard_reject == "solid_black":
        return ("Image appears to be blank (all black). "
                "Please upload a valid medical scan.")
    if hard_reject == "colorful_natural_scene":
        return ("Image appears to be a natural photograph or non-medical image. "
                "X-rays and CT scans are grayscale. "
                "Please upload a valid radiological scan.")
    if flags and not flags.get("grayscale_dominant", True):
        return ("Image contains significant color. Medical X-rays and CT scans "
                "are grayscale. Please upload a valid radiological scan.")
    if flags and not flags.get("not_overexposed", True):
        return ("Image is overexposed (too bright). "
                "Please upload a properly exposed medical scan.")
    if flags and not flags.get("not_underexposed", True):
        return ("Image is underexposed (too dark). "
                "Please upload a properly exposed medical scan.")
    if flags and not flags.get("has_edge_structure", True):
        return ("Image lacks the structural detail expected in medical scans. "
                "Please ensure the file is an unprocessed radiological scan.")
    if flags and not flags.get("not_natural_scene", True):
        return ("Image texture resembles a natural photograph rather than a "
                "medical scan. Please upload a valid X-ray or CT scan.")
    return ("Uploaded image does not appear to be a valid medical scan. "
            "Please upload an X-ray, CT scan, or MRI.")


# ══════════════════════════════════════════════════════════════════════
# MAIN PUBLIC FUNCTION
# ══════════════════════════════════════════════════════════════════════

def validate_medical_image(bgr_img: np.ndarray,
                           is_dicom: bool = False) -> dict:
    """
    Primary entry point. Called from app.py before scan type inference.

    Args:
        bgr_img:   OpenCV BGR numpy array of the uploaded image.
        is_dicom:  True if original file was .dcm — bypasses all validation.

    Returns dict with keys:
        valid (bool), status, heuristic_score, cnn_probability,
        message (if rejected), confidence (if rejected),
        low_confidence_warning (if accepted with borderline score)
    """
    start = time.time()

    # --- DICOM FAST-PASS ---
    if is_dicom:
        _log_event({
            "event": "validation_result",
            "final_decision": "accepted",
            "reason": "dicom_fast_pass",
            "processing_ms": int((time.time() - start) * 1000)
        })
        return {"valid": True, "status": "accepted", "dicom_bypass": True}

    # --- STAGE 1: HEURISTIC ---
    h_result = _compute_heuristic(bgr_img)
    h_score = h_result["score"]
    h_passed = h_result["passed"]   # True | False | None
    hard_reject = h_result.get("hard_reject")

    if h_passed is False:
        elapsed = int((time.time() - start) * 1000)
        _log_event({
            "event": "validation_result",
            "heuristic_score": h_score,
            "heuristic_passed": False,
            "hard_reject": hard_reject,
            "cnn_probability": None,
            "cnn_used": False,
            "final_decision": "rejected",
            "rejection_reason": "heuristic_failure",
            "processing_ms": elapsed
        })
        return {
            "valid": False,
            "status": "rejected",
            "reason": "non_medical_image",
            "sub_reason": hard_reject or "heuristic_failure",
            "heuristic_score": h_score,
            "cnn_probability": None,
            "message": _rejection_message(h_result.get("flags", {}), hard_reject),
            "confidence": h_score
        }

    # --- STAGE 2: CNN ---
    validator, mode = _get_validator()
    cnn_prob = None
    cnn_used = False
    confidence_warning = False

    if mode == "cnn_weights" and validator is not None and _cnn_transform is not None:
        cnn_used = True
        try:
            # Convert BGR → 3-channel grayscale for CNN
            gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
            rgb_3ch = np.stack([gray, gray, gray], axis=-1)
            tensor = _cnn_transform(rgb_3ch).unsqueeze(0)
            cnn_prob = validator.predict(tensor)
        except Exception as e:
            _log_event({"event": "cnn_inference_error", "error": str(e)})
            cnn_prob = None
            cnn_used = False
            mode = "heuristic_fallback"

    # --- CASCADE DECISION ---
    if mode == "cnn_weights" and cnn_prob is not None:
        if h_passed is True:
            # Heuristic passed — CNN acts as confirmer
            if cnn_prob >= 0.70:
                final = "accepted"
                confidence_warning = False
            elif cnn_prob >= 0.50:
                final = "accepted"
                confidence_warning = True    # Borderline — warn but accept
            else:
                final = "rejected"           # CNN overrides heuristic
        else:
            # h_passed is None (ambiguous) — CNN is the tiebreaker
            if cnn_prob >= 0.65:
                final = "accepted"
                confidence_warning = True
            else:
                final = "rejected"
    else:
        # Heuristic-only fallback (no CNN weights or CNN error)
        if h_passed is True and h_score >= 0.58:
            final = "accepted"
            confidence_warning = h_score < 0.78
        elif (h_passed is True or h_passed is None) and h_score >= 0.42:
            # Ambiguous but reasonable — accept with warning
            # CT scans often land here (0.42-0.58) due to dark backgrounds
            final = "accepted"
            confidence_warning = True
        else:
            # h_passed=False or score too low → reject
            final = "rejected"

    # --- LOG ---
    elapsed = int((time.time() - start) * 1000)
    _log_event({
        "event": "validation_result",
        "heuristic_score": h_score,
        "heuristic_passed": h_passed,
        "cnn_probability": round(cnn_prob, 4) if cnn_prob is not None else None,
        "cnn_used": cnn_used,
        "validator_mode": mode,
        "final_decision": final,
        "low_confidence_warning": confidence_warning if final == "accepted" else None,
        "processing_ms": elapsed
    })

    # --- RESPONSE ---
    if final == "accepted":
        return {
            "valid": True,
            "status": "accepted",
            "heuristic_score": h_score,
            "cnn_probability": round(cnn_prob, 3) if cnn_prob is not None else None,
            "validator_mode": mode,
            "low_confidence_warning": confidence_warning
        }
    else:
        return {
            "valid": False,
            "status": "rejected",
            "reason": "non_medical_image",
            "sub_reason": "cnn_rejection" if cnn_used else "heuristic_ambiguous_no_cnn",
            "heuristic_score": h_score,
            "cnn_probability": round(cnn_prob, 3) if cnn_prob is not None else None,
            "message": _rejection_message(h_result.get("flags", {}), None),
            "confidence": round(cnn_prob if cnn_prob is not None else h_score, 3)
        }

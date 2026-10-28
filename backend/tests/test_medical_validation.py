"""
Unit tests for the medical image validator module.
Uses synthetic images in-memory to verify acceptance of X-ray/CT patterns
and rejection of non-medical images (selfies, memes, uniform colors).
"""

import numpy as np
import cv2
import pytest
from models.medical_validator import validate_medical_image


def make_synthetic_xray(h=256, w=256):
    img = np.full((h, w), 20, dtype=np.uint8)
    cv2.rectangle(img, (w // 3, 0), (2 * w // 3, h), 180, -1)
    for y in range(40, h - 40, 30):
        cv2.line(img, (30, y), (w - 30, y), 130, 2)
    noise = np.random.normal(0, 5, (h, w)).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def make_synthetic_ct(h=256, w=256):
    img = np.full((h, w), 10, dtype=np.uint8)
    cv2.circle(img, (w // 2, h // 2), 100, 60, -1)
    cv2.circle(img, (w // 2, h // 2), 80, 30, -1)
    cv2.circle(img, (w // 2 - 20, h // 2 - 15), 15, 140, -1)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def make_synthetic_color_photo(h=256, w=256):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = 200  # Blue
    img[:, :, 1] = 50   # Green
    img[:, :, 2] = 220  # Red
    cv2.circle(img, (w // 2, h // 2), 60, (0, 255, 0), -1)
    return img


def test_dicom_bypass():
    """DICOM files should always bypass heuristic/CNN image-level checks."""
    dummy_img = make_synthetic_color_photo()
    result = validate_medical_image(dummy_img, is_dicom=True)
    assert result['valid'] is True
    assert result.get('dicom_bypass') is True


def test_synthetic_chest_xray_validation():
    """Synthetic chest X-ray should pass validation."""
    xray = make_synthetic_xray()
    result = validate_medical_image(xray, is_dicom=False)
    assert result['valid'] is True


def test_synthetic_ct_validation():
    """Synthetic CT scan should pass validation."""
    ct = make_synthetic_ct()
    result = validate_medical_image(ct, is_dicom=False)
    assert result['valid'] is True


def test_color_photo_rejection():
    """Colorful natural image/selfie should be rejected as non-medical."""
    photo = make_synthetic_color_photo()
    result = validate_medical_image(photo, is_dicom=False)
    assert result['valid'] is False
    assert result.get('reason') == 'non_medical_image'


def test_solid_black_rejection():
    """Solid black image should be rejected."""
    black = np.zeros((256, 256, 3), dtype=np.uint8)
    result = validate_medical_image(black, is_dicom=False)
    assert result['valid'] is False

"""
Medical Validator Test Matrix
=============================
Tests the heuristic + CNN validation pipeline with synthetic images.
Uses generated test images (no external dependencies).
"""

import sys
import os
import numpy as np
import cv2

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from models.medical_validator import validate_medical_image, _compute_heuristic


def make_xray_like(h=512, w=512):
    """Synthetic chest X-ray-like image: grayscale, bimodal histogram, structured edges."""
    img = np.zeros((h, w), dtype=np.uint8)
    # Dark background (lung area)
    img[:] = 20
    # Bright center (mediastinum)
    cv2.rectangle(img, (w//3, 0), (2*w//3, h), 180, -1)
    # Ribcage-like horizontal bars
    for y in range(50, h-50, 40):
        cv2.line(img, (50, y), (w-50, y), 140, 2)
    # Some noise
    noise = np.random.normal(0, 8, (h, w)).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    # Gaussian blur for realism
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def make_ct_like(h=512, w=512):
    """Synthetic CT-like: grayscale, circular structures, high dynamic range."""
    img = np.zeros((h, w), dtype=np.uint8)
    img[:] = 10
    cv2.circle(img, (w//2, h//2), 200, 60, -1)  # skull-like ring
    cv2.circle(img, (w//2, h//2), 170, 30, -1)  # inner
    cv2.circle(img, (w//2-50, h//2-30), 30, 150, -1)  # hemorrhage-like bright spot
    noise = np.random.normal(0, 5, (h, w)).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def make_bone_xray(h=512, w=300):
    """Synthetic bone X-ray: narrow aspect, grayscale, bone-like structure."""
    img = np.full((h, w), 40, dtype=np.uint8)
    # Bone shaft
    cv2.rectangle(img, (w//3, 50), (2*w//3, h-50), 180, -1)
    # Joint area
    cv2.ellipse(img, (w//2, 70), (60, 40), 0, 0, 360, 200, -1)
    noise = np.random.normal(0, 10, (h, w)).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def make_selfie(h=512, w=512):
    """Synthetic selfie: colorful, smooth gradients, natural scene texture."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Skin tone + color gradient
    img[:, :, 0] = np.linspace(100, 180, w).astype(np.uint8)  # blue channel
    img[:, :, 1] = np.linspace(120, 200, w).astype(np.uint8)  # green
    img[:, :, 2] = np.linspace(150, 230, w).astype(np.uint8)  # red (warm)
    # Face-like circle
    cv2.circle(img, (w//2, h//2), 120, (150, 180, 220), -1)
    # Add noise
    noise = np.random.normal(0, 15, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def make_meme(h=400, w=600):
    """Synthetic meme: colorful, high texture, text-like patterns."""
    img = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)
    # Bold text-like rectangles
    cv2.rectangle(img, (0, 0), (w, 60), (255, 255, 255), -1)
    cv2.rectangle(img, (0, h-60), (w, h), (255, 255, 255), -1)
    # Colorful splotches
    cv2.circle(img, (100, 200), 80, (0, 0, 255), -1)
    cv2.circle(img, (400, 200), 60, (0, 255, 0), -1)
    return img


def make_landscape(h=400, w=600):
    """Synthetic landscape: colorful, smooth, gradient sky + textured ground."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Sky gradient (blue)
    for y in range(h//2):
        ratio = y / (h//2)
        img[y, :] = (int(200 - 80*ratio), int(150 - 50*ratio), int(50 + 30*ratio))
    # Ground (green/brown)
    for y in range(h//2, h):
        ratio = (y - h//2) / (h//2)
        img[y, :] = (int(30 + 20*ratio), int(100 + 50*ratio), int(50 + 40*ratio))
    noise = np.random.normal(0, 10, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def make_solid_black(h=256, w=256):
    return np.zeros((h, w, 3), dtype=np.uint8)


def make_solid_white(h=256, w=256):
    return np.full((h, w, 3), 255, dtype=np.uint8)


def make_grayscale_photo(h=512, w=512):
    """Grayscale photo-like: BW but with natural scene texture (high LBP variance)."""
    img = np.zeros((h, w), dtype=np.uint8)
    # Natural scene-like varied texture
    for i in range(0, h, 32):
        for j in range(0, w, 32):
            val = np.random.randint(30, 220)
            img[i:i+32, j:j+32] = val
    noise = np.random.normal(0, 30, (h, w)).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def make_tiny_image(h=32, w=32):
    """Too small to be a real medical scan."""
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


# ══════════════════════════════════════════════════════════════════════
# TEST RUNNER
# ══════════════════════════════════════════════════════════════════════

def run_test(name, img, expected_valid, is_dicom=False):
    """Run a single test case."""
    result = validate_medical_image(img, is_dicom=is_dicom)
    actual = result.get('valid', False)
    status = "✅ PASS" if actual == expected_valid else "❌ FAIL"
    
    h_score = result.get('heuristic_score', result.get('dicom_bypass', '-'))
    cnn = result.get('cnn_probability', '-')
    mode = result.get('validator_mode', result.get('status', 'bypass'))
    reason = result.get('sub_reason', result.get('reason', '-'))
    
    print(f"  {status} | {name:<30s} | expected={expected_valid} actual={actual} | heuristic={h_score} cnn={cnn} mode={mode} reason={reason}")
    return actual == expected_valid


def main():
    print("=" * 100)
    print("MEDICAL VALIDATOR TEST MATRIX")
    print("=" * 100)
    
    tests = [
        # ── SHOULD ACCEPT ──
        ("Chest X-ray (synthetic)",      make_xray_like(),           True),
        ("CT scan (synthetic)",          make_ct_like(),             True),
        ("Bone X-ray (synthetic)",       make_bone_xray(),           True),
        ("DICOM bypass",                 make_selfie(),              True),  # with is_dicom=True
        
        # ── SHOULD REJECT ──
        ("Selfie/portrait (color)",      make_selfie(),              False),
        ("Meme (colorful, textured)",    make_meme(),                False),
        ("Landscape photo",              make_landscape(),           False),
        ("Solid black image",            make_solid_black(),         False),
        ("Solid white image",            make_solid_white(),         False),
        
        # ── BORDERLINE ──
        ("Grayscale photo (BW scene)",   make_grayscale_photo(),     None),  # Could go either way
        ("Tiny image (32x32)",           make_tiny_image(),          False),
    ]
    
    passed = 0
    failed = 0
    borderline = 0
    
    print("\n── SHOULD ACCEPT ──")
    for name, img, expected in tests[:3]:
        if run_test(name, img, expected):
            passed += 1
        else:
            failed += 1
    
    # Special: DICOM bypass test
    if run_test("DICOM bypass (selfie w/ .dcm)", make_selfie(), True, is_dicom=True):
        passed += 1
    else:
        failed += 1
    
    print("\n── SHOULD REJECT ──")
    for name, img, expected in tests[4:9]:
        if run_test(name, img, expected):
            passed += 1
        else:
            failed += 1
    
    print("\n── BORDERLINE (document result, no pass/fail) ──")
    for name, img, expected in tests[9:]:
        if expected is None:
            result = validate_medical_image(img)
            actual = result.get('valid', False)
            h_score = result.get('heuristic_score', '-')
            print(f"  ⚠️  INFO | {name:<30s} | valid={actual} | heuristic={h_score}")
            borderline += 1
        else:
            if run_test(name, img, expected):
                passed += 1
            else:
                failed += 1
    
    print("\n" + "=" * 100)
    print(f"RESULTS: {passed} passed, {failed} failed, {borderline} borderline")
    print("=" * 100)
    
    # Also run raw heuristic on chest X-ray to check LBP variance
    print("\n── HEURISTIC FLAG DETAIL (Chest X-ray) ──")
    h = _compute_heuristic(make_xray_like())
    for k, v in h.get('flags', {}).items():
        print(f"  {'✓' if v else '✗'} {k}")
    for k, v in h.get('details', {}).items():
        print(f"    {k}: {v}")
    
    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

"""
Pytest configuration and shared fixtures for MediScan AI Backend.
Includes test client setup and synthetic in-memory image generators.
"""

import os
import sys
import io
import shutil
import tempfile
import pytest
import numpy as np
import cv2
from PIL import Image

# Ensure backend directory is in sys.path
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app import app as flask_app


@pytest.fixture(scope="session")
def app():
    """Create and configure the Flask application instance for testing."""
    test_upload_dir = tempfile.mkdtemp(prefix="mediscan_test_uploads_")
    flask_app.config['TESTING'] = True
    flask_app.config['UPLOAD_FOLDER'] = test_upload_dir

    yield flask_app

    # Teardown temporary directory
    if os.path.exists(test_upload_dir):
        shutil.rmtree(test_upload_dir, ignore_errors=True)


@pytest.fixture
def client(app):
    """Flask test client."""
    return app.test_client()


@pytest.fixture
def make_png_bytes():
    """Helper factory to convert numpy array or PIL image to PNG in-memory bytes."""
    def _create(arr_or_img):
        buf = io.BytesIO()
        if isinstance(arr_or_img, np.ndarray):
            if arr_or_img.ndim == 2:
                img = Image.fromarray(arr_or_img, mode='L')
            else:
                img = Image.fromarray(cv2.cvtColor(arr_or_img, cv2.COLOR_BGR2RGB))
        else:
            img = arr_or_img
        img.save(buf, format='PNG')
        buf.seek(0)
        return buf
    return _create


@pytest.fixture
def valid_chest_xray_bytes(make_png_bytes):
    """
    Synthetic chest X-ray image that satisfies the medical validator and scan router.
    Grayscale, symmetric, lung fields, mediastinum.
    """
    h, w = 256, 256
    img = np.full((h, w), 25, dtype=np.uint8)
    # Bright center spine/mediastinum
    cv2.rectangle(img, (w // 3, 0), (2 * w // 3, h), 160, -1)
    # Left and right dark lung fields with ribs
    for y in range(40, h - 40, 30):
        cv2.line(img, (20, y), (w - 20, y), 120, 2)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return make_png_bytes(bgr)


@pytest.fixture
def valid_bone_xray_bytes(make_png_bytes):
    """
    Synthetic bone radiograph (narrow aspect, bone structure).
    """
    h, w = 300, 160
    img = np.full((h, w), 35, dtype=np.uint8)
    # Bone shaft
    cv2.rectangle(img, (w // 3, 30), (2 * w // 3, h - 30), 190, -1)
    # Joint head
    cv2.circle(img, (w // 2, 50), 30, 210, -1)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return make_png_bytes(bgr)


@pytest.fixture
def tiny_image_bytes(make_png_bytes):
    """Image with resolution below minimum requirement (16x16 < 32x32)."""
    img = Image.new('RGB', (16, 16), color=(100, 100, 100))
    return make_png_bytes(img)


@pytest.fixture
def corrupted_image_bytes():
    """Corrupted non-image bytes."""
    return io.BytesIO(b"NOT_A_VALID_IMAGE_DATA_STREAM")


@pytest.fixture
def non_medical_image_bytes(make_png_bytes):
    """Vibrant colorful synthetic image (should fail medical validation)."""
    h, w = 200, 200
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = 220  # Strong Blue
    img[:, :, 1] = 40   # Low Green
    img[:, :, 2] = 230  # Strong Red
    cv2.circle(img, (100, 100), 50, (0, 255, 0), -1)
    return make_png_bytes(img)

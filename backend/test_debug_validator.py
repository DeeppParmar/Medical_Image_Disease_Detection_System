import sys, os, numpy as np, cv2
sys.path.insert(0, '.')
from models.medical_validator import _compute_heuristic

# Exact CT from test_medical_validator.py
img = np.zeros((512, 512), dtype=np.uint8); img[:] = 10
cv2.circle(img, (256, 256), 200, 60, -1)
cv2.circle(img, (256, 256), 170, 30, -1)
cv2.circle(img, (256, 226), 30, 150, -1)
noise = np.random.normal(0, 5, (512, 512)).astype(np.float32)
img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
ct = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

h = _compute_heuristic(ct)
print('Score:', h['score'])
print('Passed:', h['passed'])
print('CT detected:', h.get('ct_detected', False))
print()
for k,v in h['flags'].items():
    print(' %s %s' % ('Y' if v else 'N', k))
print()
for k,v in h['details'].items():
    print(' %s: %s' % (k, v))

# Check aspect and corners manually
gray = cv2.cvtColor(ct, cv2.COLOR_BGR2GRAY)
h_px, w_px = gray.shape
aspect = w_px / h_px
corner_size = max(h_px // 10, 8)
corners = np.concatenate([
    gray[:corner_size, :corner_size].ravel(),
    gray[:corner_size, -corner_size:].ravel(),
    gray[-corner_size:, :corner_size].ravel(),
    gray[-corner_size:, -corner_size:].ravel(),
])
dark_corner_ratio = float((corners < 30).mean())
print()
print('Aspect:', aspect)
print('Corner size:', corner_size)
print('Dark corner ratio (<30):', dark_corner_ratio)
print('Is grayscale:', h['flags'].get('grayscale_dominant', False))
print('Dynamic range:', h['details'].get('dynamic_range', 0))

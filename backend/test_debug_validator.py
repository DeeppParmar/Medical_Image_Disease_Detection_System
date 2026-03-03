import sys, os, numpy as np, cv2
sys.path.insert(0, '.')
from models.medical_validator import validate_medical_image

# Exact synthetic images from test matrix
# Chest X-ray like
img = np.zeros((512, 512), dtype=np.uint8); img[:] = 20
cv2.rectangle(img, (170, 0), (340, 512), 180, -1)
for y in range(50, 462, 40):
    cv2.line(img, (50, y), (462, y), 140, 2)
noise = np.random.normal(0, 8, (512, 512)).astype(np.float32)
img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
img = cv2.GaussianBlur(img, (5, 5), 0)
xray = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
r = validate_medical_image(xray)
print('X-RAY: valid=%s score=%s' % (r.get('valid'), r.get('heuristic_score','-')))

# CT scan like  
img = np.zeros((512, 512), dtype=np.uint8); img[:] = 10
cv2.circle(img, (256, 256), 200, 60, -1)
cv2.circle(img, (256, 256), 170, 30, -1)
cv2.circle(img, (206, 226), 30, 150, -1)
noise = np.random.normal(0, 5, (512, 512)).astype(np.float32)
img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
ct = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
r = validate_medical_image(ct)
print('CT: valid=%s score=%s' % (r.get('valid'), r.get('heuristic_score','-')))

# Bone X-ray
img = np.full((512, 300), 40, dtype=np.uint8)
cv2.rectangle(img, (100, 50), (200, 462), 180, -1)
cv2.ellipse(img, (150, 70), (60, 40), 0, 0, 360, 200, -1)
noise = np.random.normal(0, 10, (512, 300)).astype(np.float32)
img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
bone = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
r = validate_medical_image(bone)
print('BONE: valid=%s score=%s' % (r.get('valid'), r.get('heuristic_score','-')))

# Selfie
img = np.zeros((512, 512, 3), dtype=np.uint8)
img[:, :, 0] = np.linspace(100, 180, 512).astype(np.uint8)
img[:, :, 1] = np.linspace(120, 200, 512).astype(np.uint8)
img[:, :, 2] = np.linspace(150, 230, 512).astype(np.uint8)
cv2.circle(img, (256, 256), 120, (150, 180, 220), -1)
noise = np.random.normal(0, 15, img.shape).astype(np.float32)
img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
r = validate_medical_image(img)
print('SELFIE: valid=%s score=%s reason=%s' % (r.get('valid'), r.get('heuristic_score','-'), r.get('sub_reason','-')))

# Solid black
r = validate_medical_image(np.zeros((256,256,3), dtype=np.uint8))
print('BLACK: valid=%s score=%s reason=%s' % (r.get('valid'), r.get('heuristic_score','-'), r.get('sub_reason','-')))

# Solid white
r = validate_medical_image(np.full((256,256,3), 255, dtype=np.uint8))
print('WHITE: valid=%s score=%s reason=%s' % (r.get('valid'), r.get('heuristic_score','-'), r.get('sub_reason','-')))

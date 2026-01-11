import os
from models.mura_inference import MURAPredictor

base = 'data/MURA/valid/XR_WRIST'
p = MURAPredictor()

# Get mix of positive and negative
all_studies = []
for pt in os.listdir(base):
    for s in os.listdir(os.path.join(base, pt)):
        sp = os.path.join(base, pt, s)
        if os.path.isdir(sp):
            imgs = [f for f in os.listdir(sp) if f.endswith('.png')]
            if imgs:
                all_studies.append((pt, s, os.path.join(sp, imgs[0])))

neg = [x for x in all_studies if 'negative' in x[1].lower()][:5]
pos = [x for x in all_studies if 'positive' in x[1].lower()][:5]

print('Testing NEGATIVE (Normal) samples:')
correct_neg = 0
for pt, study, img in neg:
    r = p.predict(img)
    is_correct = not r['is_abnormal']
    if is_correct:
        correct_neg += 1
    status = 'CORRECT' if is_correct else 'WRONG'
    print(f'  {status}: prob={r["abnormality_probability"]:.2f}')

print()
print('Testing POSITIVE (Abnormal) samples:')
correct_pos = 0
for pt, study, img in pos:
    r = p.predict(img)
    is_correct = r['is_abnormal']
    if is_correct:
        correct_pos += 1
    status = 'CORRECT' if is_correct else 'WRONG'
    print(f'  {status}: prob={r["abnormality_probability"]:.2f}')

print()
print(f'Results: Negative {correct_neg}/5, Positive {correct_pos}/5')
print(f'Total: {correct_neg + correct_pos}/10 = {(correct_neg + correct_pos) * 10}%')

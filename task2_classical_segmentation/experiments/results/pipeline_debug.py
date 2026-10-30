"""Debug pipeline stages to diagnose zero-egg detection failure."""
import sys
sys.path.insert(0, '.')
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Test on first image that gives 0 eggs
img_path = Path('data/dev/not_damaged_101_jpg.rf.479d12a2b41a5ec6d718dcdabd14b0c9.jpg')
img_bgr = cv2.imread(str(img_path))
h, w = img_bgr.shape[:2]
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

blurred_gray = cv2.GaussianBlur(gray, (5,5), 0)

# 1. Otsu
_, otsu = cv2.threshold(blurred_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
if np.mean(otsu > 0) > 0.65:
    otsu = cv2.bitwise_not(otsu)

# 2. Value Otsu
v = hsv[:,:,2]
bv = cv2.GaussianBlur(v, (5,5), 0)
_, v_otsu = cv2.threshold(bv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
if np.mean(v_otsu > 0) > 0.65:
    v_otsu = cv2.bitwise_not(v_otsu)

# 3. Saturation mask (low-sat = eggs)
s = hsv[:,:,1]
sat_mask = cv2.inRange(s, 0, 80)
bright_mask = cv2.inRange(v, 100, 255)
sat_egg = cv2.bitwise_and(sat_mask, bright_mask)

# 4. Combined
combined = cv2.bitwise_or(otsu, cv2.bitwise_or(v_otsu, sat_egg))

# 5. After morphological cleaning
k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, k_open, iterations=2)
cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_close, iterations=3)

# 6. Contours on cleaned
contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).copy()
for c in contours:
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    circ = (4*np.pi*area/perimeter**2) if perimeter > 0 else 0
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solid = area/hull_area if hull_area > 0 else 0
    x,y,cw,ch = cv2.boundingRect(c)
    ar = cw/ch if ch > 0 else 0
    color_ok = (0, 255, 0) if (2000 <= area <= 250000 and circ >= 0.25 and solid >= 0.65 and 0.4 <= ar <= 2.5) else (255, 0, 0)
    cv2.drawContours(contour_img, [c], -1, color_ok, 2)
    cv2.putText(contour_img, f'A:{area:.0f} C:{circ:.2f} S:{solid:.2f}', (x, y-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_ok, 1)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes[0][0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
axes[0][0].set_title('Original')
axes[0][1].imshow(otsu, cmap='gray')
axes[0][1].set_title('Otsu')
axes[0][2].imshow(v_otsu, cmap='gray')
axes[0][2].set_title('V-Otsu')
axes[0][3].imshow(sat_egg, cmap='gray')
axes[0][3].set_title('Saturation mask')
axes[1][0].imshow(combined, cmap='gray')
axes[1][0].set_title('Combined')
axes[1][1].imshow(cleaned, cmap='gray')
axes[1][1].set_title('After morphology')
axes[1][2].imshow(contour_img)
axes[1][2].set_title('Contours (green=pass, red=fail)')
# Also show channel histograms
axes[1][3].hist(gray.ravel(), bins=50, color='gray', alpha=0.7, label='Gray')
axes[1][3].hist(v.ravel(), bins=50, color='blue', alpha=0.5, label='V-channel')
axes[1][3].legend(); axes[1][3].set_title('Pixel histograms')
for ax in axes.flat:
    ax.axis('off') if ax != axes[1][3] else None
plt.suptitle(f'Pipeline Debug: {img_path.name[:40]}', fontsize=10)
plt.tight_layout()
plt.savefig('task2_classical_segmentation/experiments/results/pipeline_debug.png', dpi=80)
plt.close()
print('Saved pipeline_debug.png')
print(f'Contours found: {len(contours)}')
for c in contours:
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    circ = (4*np.pi*area/perimeter**2) if perimeter > 0 else 0
    x,y,cw,ch = cv2.boundingRect(c)
    ar = cw/ch if ch > 0 else 0
    hull_area = cv2.contourArea(cv2.convexHull(c))
    solid = area/hull_area if hull_area > 0 else 0
    print(f'  area={area:.0f} circ={circ:.3f} solid={solid:.3f} ar={ar:.2f}')

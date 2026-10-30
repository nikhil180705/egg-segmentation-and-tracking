"""Quick test script for improved classical segmenter."""
import sys
sys.path.insert(0, '.')
from task2_classical_segmentation.src.egg_segmenter import ClassicalEggSegmenter
from pathlib import Path

seg = ClassicalEggSegmenter()
imgs = sorted(Path('data/dev').glob('*.jpg'))[:15]

total_eggs = 0
zero_imgs = 0
for img_path in imgs:
    result = seg.segment_image(img_path)
    n = result['num_eggs']
    total_eggs += n
    if n == 0:
        zero_imgs += 1
    t = result['processing_time_s']
    print(f"{img_path.name[:35]:35s}: {n} eggs  {t:.3f}s")

print(f"\nTotal: {total_eggs} eggs | Zero-egg images: {zero_imgs}/{len(imgs)}")
print(f"Avg eggs/image: {total_eggs/len(imgs):.2f}")

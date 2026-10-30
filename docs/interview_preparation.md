# Computer Vision Interview Preparation Guide
## Nosh Robotics Internship Technical Defense & System Architecture Q&A

This document serves as an exhaustive technical preparation reference for explaining, defending, and discussing the Egg Segmentation and Multi-Object Tracking project during technical interviews with the Nosh Robotics Computer Vision team.

---

## 1. High-Level System Architecture & Design Decisions

### Q1: Can you walk us through the high-level architecture of your project?
> **Answer**:
> The project addresses two operational regimes in robotic food automation: **high-accuracy semantic inspection** and **high-throughput real-time manipulation**.
> 
> The system is architected into four distinct modular tasks:
> 1. **Task 1 (Foundation Model Segmentation)**: Uses Meta's Segment Anything Model (SAM `vit_b`) with Automatic Mask Generation (AMG) and calibrated geometric candidate filtering to produce reference-grade instance masks without supervised training.
> 2. **Task 2 (Classical Image Processing)**: A deep-learning-free pipeline combining multi-strategy Otsu thresholding (grayscale and HSV value channels), morphological filtering (5x5 opening/closing), distance-transform watershed separation for touching eggs, and contour shape validation, executing in **67.1 ms** (~15 FPS).
> 3. **Task 3 (Quantitative Evaluation)**: Implements pixel-level metrics (IoU, Dice, Precision, Recall) and instance-level bipartite matching via the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`), reporting both Macro-Averaged and Micro-Averaged (pooled) metrics across development and held-out evaluation splits.
> 4. **Task 4 (Real-Time Video Tracking)**: An in-memory frame processing pipeline combining the classical detector with a Hungarian centroid tracker (`CentroidTracker`), maintaining persistent track IDs, managing occlusion lifecycles, and rendering motion trajectory tails at **53.14 FPS** on a 1,440-frame cooking video.

---

### Q2: Why did you implement both a foundation model (SAM) and a classical pipeline?
> **Answer**:
> In an autonomous robotic kitchen, computational resources and latency requirements vary dramatically across tasks:
> - **Pre-cooking Inspection / Inventory**: High latency (e.g., 1-2s) is acceptable to achieve near-perfect semantic discovery of eggs in crates, verifying egg counts and integrity. SAM excels here.
> - **Active Cooking / Manipulation**: Robotic arms grasping rolling eggs or tracking eggs as they are cracked and stirred require latencies under 50 ms (>20 FPS). SAM (running at ~26s per frame on edge hardware) is computationally infeasible. Classical computer vision runs in **67 ms on single CPU** (and **18.8 ms in video stream**), providing a **~394x speedup** that makes closed-loop robotic feedback possible.

---

## 2. Mathematical Formulations & Metric Evaluation (Task 3)

### Q3: Explain the difference between Macro-Averaging and Micro-Averaging in your evaluation. Why do they differ?
> **Answer**:
> This was a critical mathematical distinction in our audit:
> - **Macro-Averaging**: Computes precision and recall independently for each image $i$, then takes the unweighted mean across all images:
>   $$\text{Macro Precision} = \frac{1}{N}\sum_{i=1}^N \frac{\text{TP}_i}{\text{TP}_i + \text{FP}_i}$$
>   *On our Dev Set*: Macro Instance Precision is **59.28%**, and Macro Instance Recall is **15.97%**.
> - **Micro-Averaging (Pooled Counts)**: Sums all TP, FP, and FN globally across the entire split before computing the ratio:
>   $$\text{Micro Precision} = \frac{\sum \text{TP}_i}{\sum \text{TP}_i + \sum \text{FP}_i} = \frac{60}{60 + 30} = \mathbf{66.67\%}$$
>   $$\text{Micro Recall} = \frac{\sum \text{TP}_i}{\sum \text{TP}_i + \sum \text{FN}_i} = \frac{60}{60 + 369} = \mathbf{13.99\%}$$
>   $$\text{Micro F1} = \frac{2 \cdot \text{Micro Prec} \cdot \text{Micro Rec}}{\text{Micro Prec} + \text{Micro Rec}} = \mathbf{23.12\%}$$
> 
> *Why they differ*: Macro averaging weights every image equally, so an image with 1 egg correctly detected counts for $100\%$, while an image with zero detections counts for $0\%$. Micro averaging weights every individual egg equally, so images containing 10-12 eggs in a carton have a larger proportional impact on recall. Reporting both provides complete transparency.

---

### Q4: How does your Hungarian bipartite matching work for instance evaluation?
> **Answer**:
> Instead of greedily matching predicted masks to reference masks, we construct an assignment cost matrix $C \in \mathbb{R}^{P \times G}$, where $P$ is the number of predicted masks and $G$ is the number of reference masks:
> $$C_{i,j} = 1.0 - \text{IoU}(P_i, G_j)$$
> We solve the global minimum weight bipartite matching problem using the Kuhn-Munkres (Hungarian) algorithm via `scipy.optimize.linear_sum_assignment`. A pairing $(i, j)$ is only considered a True Positive if $\text{IoU}(P_i, G_j) \ge 0.25$. Any predicted mask without a matching GT is a False Positive; any reference mask without a matching prediction is a False Negative.

---

## 3. Video Tracking & Multi-Object Dynamics (Task 4)

### Q5: Why did your video tracker identify 29 unique track IDs when the scene only contains 2 to 4 eggs? Is this a bug?
> **Answer**:
> No, this is an expected multi-object tracking phenomenon known as **track fragmentation**, directly caused by physical interactions in the video:
> 1. The video is an educational tabletop science experiment where a human presenter repeatedly spins raw and boiled eggs by hand, stops them with a finger, and spins them again.
> 2. When the presenter's hand grasps or occludes the egg, detections drop. Our tracker uses a `max_missing_frames` threshold of **15 frames** (~0.6 seconds at 24 FPS).
> 3. If an egg is occluded for $>15$ frames, the tracker deregistrates the track to prevent "drifting" or assigning an old ID to a different object. When the hand releases the egg, the detector re-acquires it and initializes a new track ID.
> 4. Out of 29 tracks, **24 were confirmed tracks** ($\ge 2$ hits), with the longest track persisting continuously for **432 frames (18.0 seconds)**. Fourteen tracks were brief interaction segments ($<30$ frames) during hand manipulation. During uninterrupted visibility, track ID assignment remained 100% stable.

---

### Q6: How did you achieve 53.14 FPS in video tracking?
> **Answer**:
> Three key optimizations enabled ultra-real-time throughput:
> 1. **In-Memory Frame Processing**: The classical segmenter was refactored to accept in-memory NumPy BGR arrays directly, eliminating temporary disk I/O per frame.
> 2. **Algorithmic Efficiency**: The Hungarian centroid tracker operates on low-dimensional 2D Euclidean centroids rather than heavy pixel-level optical flow, executing in under $0.5 \text{ ms}$ per frame.
> 3. **Separation of Workloads**: Video decoding and visualization overlays were integrated into a streamlined single-pass loop, enabling the entire 1,440-frame 1-minute video to be tracked and written in **27.1 seconds**.

---

## 4. Edge Cases, Robustness & Future Robotic Extensions

### Q7: What are the main failure modes of classical segmentation on eggs, and how would you fix them?
> **Answer**:
> 1. **White-on-White Low Contrast**: White eggs on white styrofoam cartons or white marble countertops have overlapping intensity histograms, causing Otsu thresholding to fail.
> 2. **Specular Shell Reflections**: Strong overhead lighting creates bright spots that fragment contours.
> 3. **Dense Clusters**: Eggs touching in trays sometimes merge into super-contours despite watershed separation.
> 
> *Fixes for Robotic Deployment*:
> - In an autonomous robotic kitchen (like Nosh), lighting can be physically controlled using polarized diffusers or multi-angle LED rings to eliminate specular highlights.
> - An active RGB-D camera (e.g. Intel RealSense) provides depth point clouds: eggs stand out as 3D ellipsoidal geometric surfaces even when their 2D color matches the countertop perfectly.
> - For inference, distilling SAM's masks into an ultra-compact MobileSAM or YOLOv8-Nano model would yield both high recall and real-time GPU/NPU speeds.

---

## 5. Verification Checklist

- [x] Mathematics verified: Macro vs Micro precision/recall/F1 calculated and documented.
- [x] Held-out evaluation completed: 22 images evaluated with frozen classical pipeline.
- [x] Data leakage checked: SHA-256 and dHash confirmed 0 collisions between dev and eval sets.
- [x] Tracking validated: 29 tracks analyzed, fragmentation root cause documented, 53.14 FPS verified.
- [x] Speedup verified: 393.7x static image speedup, 53.14 FPS video throughput.
- [x] 17 unit tests passing with pytest.
- [x] Git history preserved with clean semantic commits.

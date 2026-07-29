# Complex Phase-Encoded Vector Displacement Similarity (CPEVD)

**CPEVD** is an image evaluation and computer vision library engineered for semantic segmentation assessment. While standard pixel-matching statistics (like Dice and Intersection over Union) capture raw overlap, they often fail to evaluate structural boundary alignments accurately. 

CPEVD computes the structural fidelity of segmentation masks by transforming binary edges into directional gradient fields using complex numbers, tracking alignment coherence, and outputting an elegant, multi-stage analytical visual grid.

## Key Features

- **Boundary Structural Orientation:** Extracts Euclidean Distance Transforms (EDT) and maps them to a complex plane ($z = dx - i \cdot dy$) to capture boundary phase directions.
- **Automated Analytical Grid Plotting:** Generates complete debugging and visualization charts detailing distance maps, gradient fields, phase vectors, overlay errors, and directional coherence values ($C_{\text{norm}}$).
- **Comprehensive Metric Suite:** Combines CPEVD with classic scoring frameworks like Hausdorff Distance, Dice-Sørensen Coefficient, Jaccard Index (IoU), Accuracy, Precision, Sensitivity, and Specificity.

- ## Installation

Install the package and all of its underlying scientific dependencies seamlessly via `pip`:

```bash
pip install cpevd
```

---

## Quickstart & Usage

The code adaptively resizes arrow vectors and scales analysis parameters based on whatever dimensions your user passes in.

### 1. CPEVD Score

```python
import cpevd

# Load the user masks
gt_mask = cpevd.load_mask("my_user_gt.png")
pred_mask = cpevd.load_mask("my_user_pred.png")

# Generate the comprehensive evaluation matrix plots
pipeline_results = cpevd.plot_cpevd_intermediates(
    gt_mask=gt_mask,
    pred_mask=pred_mask,
    title="Plots"
)

# Extract specific structural alignment coefficients
print(f"CPEVD Structural Similarity Index: {pipeline_results['cpevd']:.4f}")
```
---
## Print Complete Performance Summary
```python
import cpevd

# Load the user masks
gt_mask = cpevd.load_mask("my_user_gt.png")
pred_mask = cpevd.load_mask("my_user_pred.png")
# Calculate metrics and details
results = cpevd.compute_all_metrics(gt_mask, pred_mask)
cpevd_results = cpevd.complex_phase_vector_displacement_similarity_detailed(gt_mask, pred_mask)

print("\n" + "="*50)
print("              PERFORMANCE EVALUATION")
print("="*50)
print(f" CPEVD Similarity Index : {cpevd_results['cpevd']:.4f}")
print(f" Dice Coefficient      : {results['dice']:.4f}")
print(f" Jaccard Index (IoU)   : {results['jaccard']:.4f}")
print(f" Accuracy              : {results['accuracy']:.4f}")
print(f" Precision             : {results['precision']:.4f}")
print(f" Sensitivity (Recall)  : {results['sensitivity']:.4f}")
print(f" Specificity           : {results['specificity']:.4f}")
print(f" Hausdorff Distance    : {results['hausdorff']:.2f} pixels")
print("="*50)
````
## Find average of entire test set
```python
import os
import glob
from collections import defaultdict
import cv2
import cpevd

# Define folder paths
GT_DIR = r"Path to Test GT"
PRED_DIR = r"Path to Predicted "

# Supported image extensions
IMAGE_EXTENSIONS = ('*.bmp', '*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff')

# Collect all files from GT directory
gt_files = []
for ext in IMAGE_EXTENSIONS:
    gt_files.extend(glob.glob(os.path.join(GT_DIR, ext)))

# Storage for metric totals
metrics_sum = defaultdict(float)
valid_pairs_count = 0

print("Processing image pairs...")

for gt_path in gt_files:
    filename = os.path.basename(gt_path)
    pred_path = os.path.join(PRED_DIR, filename)

    # Check if corresponding predicted mask exists
    if not os.path.exists(pred_path):
        print(f"Warning: Corresponding prediction not found for {filename}. Skipping.")
        continue

    try:
        # Load ground truth and prediction masks
        gt_mask = cpevd.load_mask(gt_path)
        pred_mask = cpevd.load_mask(pred_path)

        # Get original GT shape: gt_mask.shape gives (height, width)
        orig_h, orig_w = gt_mask.shape[:2]
        target_size = (orig_w, orig_h)  # cv2.resize expects (width, height)

        # Resize prediction mask to match the original ground truth shape
        if pred_mask.shape[:2] != (orig_h, orig_w):
            pred_mask = cv2.resize(pred_mask, target_size, interpolation=cv2.INTER_NEAREST)

        # Calculate metrics and details
        results = cpevd.compute_all_metrics(gt_mask, pred_mask)
        cpevd_results = cpevd.complex_phase_vector_displacement_similarity_detailed(gt_mask, pred_mask)

        # Accumulate metrics
        metrics_sum['cpevd'] += cpevd_results['cpevd']
        metrics_sum['dice'] += results['dice']
        metrics_sum['jaccard'] += results['jaccard']
        metrics_sum['accuracy'] += results['accuracy']
        metrics_sum['precision'] += results['precision']
        metrics_sum['sensitivity'] += results['sensitivity']
        metrics_sum['specificity'] += results['specificity']
        metrics_sum['hausdorff'] += results['hausdorff']

        valid_pairs_count += 1

    except Exception as e:
        print(f"Error processing {filename}: {e}")

# Calculate averages and print summary
if valid_pairs_count > 0:
    avg = {key: val / valid_pairs_count for key, val in metrics_sum.items()}

    print("\n" + "="*50)
    print(f"AVERAGE PERFORMANCE EVALUATION ({valid_pairs_count} Images)")
    print("="*50)
    print(f" CPEVD Similarity Index : {avg['cpevd']:.4f}")
    print(f" Dice Coefficient       : {avg['dice']:.4f}")
    print(f" Jaccard Index (IoU)    : {avg['jaccard']:.4f}")
    print(f" Accuracy               : {avg['accuracy']:.4f}")
    print(f" Precision              : {avg['precision']:.4f}")
    print(f" Sensitivity (Recall)   : {avg['sensitivity']:.4f}")
    print(f" Specificity            : {avg['specificity']:.4f}")
    print(f" Hausdorff Distance     : {avg['hausdorff']:.2f} pixels")
    print("="*50)
else:
    print("No matching image pairs found to process.")
````




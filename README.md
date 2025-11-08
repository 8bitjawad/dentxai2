# DentXAI: Explainable AI for Dental Caries Severity Classification

## 🦷 Overview

**DentXAI** is an advanced deep learning framework that combines the power of YOLOv8 with cutting-edge Explainable AI (XAI) techniques to classify and assess the severity of dental caries in intraoral images. The system goes beyond simple detection by providing transparent, interpretable insights into model predictions, making it valuable for both dental professionals and AI researchers.

### Key Classification Categories
- **Low Severity Caries** - Early-stage cavities requiring minimal intervention
- **Moderate Severity Caries** - Progressive decay requiring standard treatment
- **Severe Caries** - Advanced decay requiring immediate attention

### Why Explainability Matters

In medical imaging, understanding *why* a model makes a particular prediction is as crucial as the prediction itself. DentXAI addresses this need by integrating two powerful XAI techniques that reveal the model's decision-making process for each severity level, fostering trust and enabling clinical validation.

> **⚠️ Current Status**: This model is in active development. Due to limited training data, the current accuracy is not yet suitable for clinical deployment. We are continuously improving the model through data augmentation and expanded datasets. See [Future Improvements](#-future-improvements) for our roadmap.

---

## 🎯 Features

- **🤖 Multi-Class Severity Detection**: Leverages YOLOv8 architecture to classify caries into three severity levels
- **🔍 LRP-based Saliency Maps**: Generates class-specific Layer-wise Relevance Propagation visualizations using Easy-Explain
- **🧩 Occlusion Sensitivity Analysis**: Identifies critical image regions for each severity class through systematic masking
- **📊 Comparative XAI Visualization**: Side-by-side comparison of XAI outputs across all severity levels
- **🎨 Multi-Class Heatmap Generation**: Separate and combined heatmaps showing severity-specific features
- **⚕️ Clinical Transparency**: Ensures the model focuses on clinically relevant anatomical features for each severity level

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- CUDA-compatible GPU (recommended for training and inference)

### Setup

```bash
# Clone the repository
git clone https://github.com/8bitjawad/dentxai_copy.git
cd dentxai_copy

# Install dependencies
pip install -r requirements.txt
```

### Required Dependencies

The project requires the following key libraries:
- `ultralytics` (YOLOv8)
- `torch` and `torchvision`
- `opencv-python`
- `matplotlib`
- `numpy`
- `easy-explain` (for LRP-based saliency maps)
- `Pillow` (for image processing)

---

## 📖 Usage

### 1. Dataset Preparation

Organize your dental intraoral images into the following structure:

```
dataset/
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

### 2. Configuration

Edit the `data.yaml` file to specify your dataset paths and class labels:

```yaml
path: ./dataset
train: train/images
val: val/images

nc: 3  # number of classes
names: ['low', 'moderate', 'severe']
```

### 3. Model Training

```bash
python train.py --data data.yaml --cfg yolov8n.yaml --weights yolov8n.pt --epochs 100
```

**Training Parameters:**
- `--data`: Path to data configuration file
- `--cfg`: Model configuration file
- `--weights`: Pretrained weights for transfer learning
- `--epochs`: Number of training epochs

### 4. Generate XAI Explanations

#### Easy-Explain: LRP-Based Saliency Maps (Multi-Class)

```python
from ultralytics import YOLO
from easy_explain import YOLOv8LRP
import cv2
import torch

# Load model and image
model_path = "path/to/best.pt"
model = YOLO(model_path)
image_path = "path/to/test_image.jpg"

# Prepare image tensor
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (640, 640))
image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

# Initialize LRP explainer
lrp = YOLOv8LRP(model.model, power=2, eps=1, device='cpu')

# Generate explanations for ALL severity classes
for class_idx in range(len(model.names)):
    class_name = model.names[class_idx]
    
    # Generate class-specific explanation
    explanation = lrp.explain(image_tensor, cls=class_idx, contrastive=False).cpu()
    
    # Visualize
    lrp.plot_explanation(
        frame=image_tensor,
        explanation=explanation,
        contrastive=False,
        cmap='seismic',
        title=f'XAI Explanation: {class_name.upper()} Severity'
    )
```

#### Occlusion Sensitivity: Multi-Class Analysis

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

# Load model and image
model = YOLO("path/to/best.pt")
image = np.array(Image.open("path/to/test_image.jpg").convert('RGB'))

# Get original predictions per class
orig_results = model(image)[0]
orig_conf_per_class = {}
for i in range(len(model.names)):
    class_boxes = [box for box in orig_results.boxes if int(box.cls) == i]
    orig_conf_per_class[i] = max([float(box.conf) for box in class_boxes]) if class_boxes else 0.0

# Occlusion parameters
patch_size = (50, 50)
step_size = 25
height, width, _ = image.shape

# Create heatmap for each class
heatmaps = {i: np.zeros((height, width), dtype=np.float32) for i in range(len(model.names))}

# Systematic occlusion
for y in range(0, height - patch_size[1] + 1, step_size):
    for x in range(0, width - patch_size[0] + 1, step_size):
        # Occlude region
        occluded = image.copy()
        region = occluded[y:y+patch_size[1], x:x+patch_size[0]]
        blurred = cv2.GaussianBlur(region, (21, 21), 0)
        occluded[y:y+patch_size[1], x:x+patch_size[0]] = blurred
        
        # Measure confidence drop for each class
        occl_results = model(occluded)[0]
        for cls_idx in range(len(model.names)):
            class_boxes = [box for box in occl_results.boxes if int(box.cls) == cls_idx]
            occl_conf = max([float(box.conf) for box in class_boxes]) if class_boxes else 0.0
            diff = float(orig_conf_per_class[cls_idx] - occl_conf)
            heatmaps[cls_idx][y:y+patch_size[1], x:x+patch_size[0]] += diff

# Visualize class-specific heatmaps
for cls_idx in range(len(model.names)):
    class_name = model.names[cls_idx]
    normalized_heatmap = (heatmaps[cls_idx] - heatmaps[cls_idx].min()) / (heatmaps[cls_idx].max() - heatmaps[cls_idx].min())
    
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.imshow(normalized_heatmap, cmap='jet', alpha=0.5)
    plt.title(f'Occlusion Heatmap: {class_name.upper()} Severity')
    plt.axis('off')
    plt.colorbar(label='Sensitivity')
    plt.show()
```

---

## 🔬 Explainable AI Techniques

### 1. Layer-wise Relevance Propagation (LRP) via Easy-Explain

**Purpose**: Visualizes which regions of the intraoral image the model focuses on when predicting each severity level.

**How It Works**:
- Propagates relevance scores backward through the network layers
- Assigns importance values to each pixel based on contribution to the class prediction
- Generates class-specific heat maps overlaid on the original image

**Multi-Class Interpretation**:
- **Strong, focused activation** for the correct severity class → Model is confident and correctly identifies severity-specific features
- **Weak, diffuse activation** for incorrect classes → Model properly distinguishes between severity levels
- **Similar activations across all classes** → Model may need better training data or architectural improvements

**Clinical Value**:
- Validates that the model attends to tooth structures and lesion characteristics
- Shows severity-specific feature discrimination (e.g., depth of decay, extent of damage)
- Helps identify if the model correctly differentiates between early, moderate, and advanced caries

**Example Output**:

```
Input: Intraoral photo → Model → Predictions:
                                   - Low: 15% confidence
                                   - Moderate: 78% confidence ✓
                                   - Severe: 7% confidence
                              ↓
                    Three Saliency Maps Generated:
                    - Low severity: Weak, diffuse activation
                    - Moderate severity: Strong, focused on lesion ✓
                    - Severe severity: Weak, peripheral activation
```

**Interpretation Guide**:
- 🔴 **Red/Hot colors**: High positive relevance (strongly supports this severity class)
- 🔵 **Blue/Cool colors**: Negative relevance (evidence against this severity class)
- ✅ **Expected pattern**: Strongest activation for the correct severity level, concentrated on the carious region
- ⚠️ **Warning sign**: Similar heatmaps across all severity classes may indicate insufficient training data

### 2. Occlusion Sensitivity Analysis (Multi-Class)

**Purpose**: Systematically determines which parts of the image are critical for classifying each severity level.

**How It Works**:
1. Slides a masking window across the entire image
2. For each position, occludes (blurs) that region
3. Re-runs detection and records confidence changes **for each severity class**
4. Creates separate sensitivity maps showing impact on each severity prediction

**Multi-Class Interpretation**:
- **Class-specific sensitivity patterns** → Different regions matter for different severity levels
- **High sensitivity for detected class only** → Model correctly identifies severity-specific features
- **Overlapping high-sensitivity regions** → Model may struggle to differentiate severity levels

**Clinical Value**:
- Confirms the model relies on actual caries characteristics (color, depth, extent) rather than artifacts
- Shows how severity assessment changes when different regions are hidden
- Identifies the minimal region necessary for accurate severity classification

**Example Output**:

```
Original Predictions:
- Low: 20% | Moderate: 75% | Severe: 5%

Occlude Region A (shallow discoloration):
- Low: 5% (↓15%) → HIGH sensitivity for LOW
- Moderate: 70% (↓5%) → LOW sensitivity for MODERATE
- Severe: 5% (no change) → NO sensitivity for SEVERE

Occlude Region B (deep cavity):
- Low: 18% (↓2%) → LOW sensitivity
- Moderate: 30% (↓45%) → CRITICAL for MODERATE ✓
- Severe: 3% (↓2%) → LOW sensitivity
```

**Interpretation Guide**:
- 🟥 **High sensitivity (bright red)**: Critical region for this severity class
- 🟨 **Moderate sensitivity (yellow)**: Supporting contextual information
- 🟩 **Low sensitivity (green/dark)**: Not important for this severity classification
- 🎯 **Ideal pattern**: Each severity class should show sensitivity to different anatomical features (surface vs. deep lesions)

### Comparing Both Techniques for Multi-Class Analysis

| Aspect | LRP (Easy-Explain) | Occlusion Sensitivity |
|--------|-------------------|----------------------|
| **Computation** | Gradient-based (fast) | Perturbation-based (slower) |
| **Granularity** | Pixel-level | Patch-level |
| **Multi-Class** | Shows what model "sees" per class | Shows what model "needs" per class |
| **Interpretation** | Positive/negative evidence | Confidence drop magnitude |
| **Best For** | Quick severity comparison | Robust validation of class-specific features |
| **Clinical Use** | Initial severity screening | Detailed verification of diagnostic criteria |

### Why Both Techniques Together?

Using both XAI methods provides complementary insights:

1. **LRP shows feature attribution**: "The model thinks this dark region indicates moderate severity"
2. **Occlusion validates importance**: "Hiding that region drops moderate severity confidence by 45%"

If LRP highlights a region but occlusion shows low sensitivity (or vice versa), it may indicate:
- Model uncertainty
- Spurious correlations
- Need for additional training data

---

## 📊 Example Outputs

### LRP Multi-Class Comparison

**Low Severity Heatmap**
<img width="800" alt="LRP Low Severity" src="https://github.com/user-attachments/assets/example-lrp-low.png" />
*Weak activation pattern - model correctly identifies this is not a low-severity case.*

**Moderate Severity Heatmap**
<img width="800" alt="LRP Moderate Severity" src="https://github.com/user-attachments/assets/example-lrp-moderate.png" />
*Strong, focused activation on the carious lesion - model confidently predicts moderate severity.*

**Severe Severity Heatmap**
<img width="800" alt="LRP Severe Severity" src="https://github.com/user-attachments/assets/example-lrp-severe.png" />
*Minimal activation - model correctly rules out severe classification.*

### Occlusion Multi-Class Sensitivity

**Side-by-Side Comparison**
<img width="1200" alt="Occlusion All Classes" src="https://github.com/user-attachments/assets/example-occlusion-comparison.png" />
*Each severity class shows different sensitivity patterns. Moderate severity (center) has highest sensitivity at the lesion site, confirming correct classification.*

**Combined RGB Heatmap**
<img width="800" alt="Combined Occlusion Map" src="https://github.com/user-attachments/assets/example-combined-rgb.png" />
*RGB composite where Red=Low, Green=Moderate, Blue=Severe. Green dominance at the lesion validates moderate severity prediction.*

---

## 🛠️ Advanced Usage

### Batch Processing with Multi-Class XAI

```python
import os
from pathlib import Path

# Process multiple images
image_dir = Path("test_images")
output_dir = Path("xai_results")
output_dir.mkdir(exist_ok=True)

for img_path in image_dir.glob("*.jpg"):
    print(f"Processing {img_path.name}...")
    
    # Load and prepare image
    image = cv2.imread(str(img_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (640, 640))
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
    
    # Generate LRP explanations for all classes
    for cls_idx in range(len(model.names)):
        class_name = model.names[cls_idx]
        explanation = lrp.explain(image_tensor, cls=cls_idx, contrastive=False).cpu()
        
        # Save class-specific LRP
        save_path = output_dir / f"{img_path.stem}_lrp_{class_name}.png"
        lrp.plot_explanation(
            frame=image_tensor,
            explanation=explanation,
            contrastive=False,
            cmap='seismic',
            title=f'{class_name.upper()}: {img_path.name}'
        )
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"  ✓ Saved LRP explanations for all classes")
```

### Confidence-Weighted Visualization

```python
# Show only high-confidence class explanations
results = model(image)[0]
confidence_threshold = 0.3

for box in results.boxes:
    cls_idx = int(box.cls)
    conf = float(box.conf)
    
    if conf > confidence_threshold:
        class_name = model.names[cls_idx]
        print(f"Generating XAI for {class_name} (conf: {conf:.2f})")
        
        # Generate explanation only for confident predictions
        explanation = lrp.explain(image_tensor, cls=cls_idx, contrastive=False).cpu()
        lrp.plot_explanation(
            frame=image_tensor,
            explanation=explanation,
            contrastive=False,
            cmap='seismic',
            title=f'{class_name.upper()} - Confidence: {conf:.2%}'
        )
```

---

## 📚 Understanding Multi-Class Results

### What Makes a Good Multi-Class Explanation?

1. **Class Separation**: Different severity levels should show distinct heatmap patterns
2. **Anatomical Accuracy**: XAI highlights should correspond to lesion depth and extent
3. **Confidence Alignment**: Strongest XAI activation should match the highest-confidence class
4. **Clinical Consistency**: Similar severity cases should show similar explanation patterns
5. **Progressive Features**: Low → Moderate → Severe should show progressively deeper/larger affected regions

### Red Flags in Multi-Class XAI Outputs

- ⚠️ All three severity classes show identical heatmaps → Model hasn't learned severity differentiation
- ⚠️ Highest activation doesn't match the predicted class → Model uncertainty or poor training
- ⚠️ Saliency concentrated on image borders, watermarks, or non-dental regions → Dataset bias
- ⚠️ High sensitivity to background or teeth unaffected by caries → Spurious correlations
- ⚠️ No correlation between lesion location and explanation highlights → Model not using clinical features

### Interpreting Disagreements Between XAI Methods

If LRP and Occlusion show different critical regions:

1. **LRP highlights Region A, Occlusion highlights Region B**
   - Possible cause: LRP shows initial attention, occlusion shows what's actually necessary
   - Action: Trust occlusion more for clinical validation

2. **Both highlight same region but for different classes**
   - Possible cause: Insufficient training data to distinguish severity
   - Action: Collect more labeled examples, especially borderline cases

3. **LRP shows strong activation, but occlusion shows low sensitivity**
   - Possible cause: Model using texture/color but not spatial structure
   - Action: Review if model is learning clinically meaningful features

---

## ⚠️ Current Limitations

### Model Accuracy
- **Training Data**: The current model is trained on a limited dataset of intraoral photographs
- **Performance**: Accuracy is not yet suitable for clinical deployment
- **Variability**: Performance may vary significantly across different imaging conditions and patient populations
- **Severity Classification**: Distinguishing between adjacent severity levels (e.g., low vs. moderate) is particularly challenging

### Known Issues
- Similar XAI heatmaps across severity classes indicate the model needs more diverse training data
- Occlusion sensitivity may show overlapping critical regions for different severity levels
- Model may struggle with borderline cases between severity classifications

### Appropriate Use
- ✅ Research and educational purposes
- ✅ Algorithm development and XAI methodology testing
- ✅ Dataset annotation assistance (with expert oversight)
- ❌ **NOT for clinical diagnosis or treatment decisions**
- ❌ **NOT as a replacement for professional dental examination**

---

## 🔮 Future Improvements

### Short-Term Goals
1. **Dataset Expansion**: Collect and annotate significantly more intraoral images across all severity levels
2. **Data Augmentation**: Implement advanced augmentation techniques to improve model generalization
3. **Balanced Training**: Ensure equal representation of low, moderate, and severe cases
4. **Hyperparameter Tuning**: Optimize model architecture and training parameters for severity classification

### Medium-Term Goals
1. **X-Ray Integration**: Expand to radiographic images (bitewing, periapical) for more accurate severity assessment
2. **Multi-Modal Learning**: Combine intraoral photos and X-rays for comprehensive diagnosis
3. **Clinical Validation**: Partner with dental professionals for real-world testing and validation
4. **Ensemble Methods**: Combine multiple models to improve classification robustness

### Long-Term Vision
1. **Real-Time Detection**: Deploy lightweight models for chairside use
2. **Treatment Planning Integration**: Link severity classification to evidence-based treatment recommendations
3. **Longitudinal Tracking**: Monitor caries progression over time with temporal XAI
4. **3D Imaging Support**: Extend to CBCT and intraoral scanner data for volumetric analysis

---

## 🤝 Contributing

We welcome contributions to improve DentXAI! Priority areas include:

- **Dataset Contribution**: Annotated dental images (especially X-rays)
- **XAI Techniques**: Implementation of additional explainability methods
- **Clinical Validation**: Collaboration with dental professionals
- **Code Optimization**: Performance improvements and bug fixes

Please open an issue or submit a pull request on GitHub.
---

## 📞 Contact

For questions about multi-class XAI implementations or clinical applications:

- **GitHub**: [@8bitjawad](https://github.com/8bitjawad)
- **Issues**: [GitHub Issues](https://github.com/8bitjawad/dentxai_copy/issues)

---

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics for the robust detection framework
- Easy-Explain and Occlusion Detection library for LRP-based visualization
- The dental imaging research community for open datasets and clinical guidance
- All contributors who have helped improve this project

---
## ⚖️ Disclaimer

**IMPORTANT**: This software is provided for research and educational purposes only. It is NOT intended for clinical use, medical diagnosis, or treatment decisions. Always consult qualified dental professionals for oral health concerns. The developers assume no liability for any use of this software in clinical settings.

---

**Made with 🦷 for transparent and trustworthy dental AI**

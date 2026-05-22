# Real-Time Multi-Class Candy Object Detection Pipeline

An end-to-end computer vision pipeline developed to detect, localize, and classify 11 distinct types of candies from image data. Built using the state-of-the-art **YOLOv8 nano** architecture, this project demonstrates custom data preparation, automated configuration generation, and highly efficient transfer learning tailored for lightweight deployment scenarios.

## 🚀 Performance Summary
* **Inference Speed:** ~1.4ms per image on an NVIDIA Tesla T4 GPU (highly optimized for real-time inference latency).
* **Overall Accuracy (mAP50):** 98.6%
* **Strict Localization & Classification (mAP50-95):** 88.9%

---

## 🛠️ Tech Stack & Key Architectures
* **Core Framework:** Python 3.11, PyTorch (v2.5.1+cu124)
* **Model Framework:** Ultralytics YOLOv8 (Nano variant: `yolov8n.pt`)
* **Infrastructure:** CUDA 12.4 utilizing an NVIDIA Tesla T4 GPU
* **Data Pipelines:** YAML configuration parsing, custom automated validation split generation.

---

## 📋 Custom Dataset Architecture & Preprocessing
The custom dataset features a severe class imbalance and specialized item packaging, containing 11 target classes:
* *Class Directory:* `MMs_peanut`, `MMs_regular`, `airheads`, `gummy_worms`, `milky_way`, `nerds`, `skittles`, `snickers`, `starbust`, `three_musketeers`, `twizzlers`.

### Data Preprocessing & Validation Flow
1. **Automated Validation Split:** Initial dataset containing annotated files was systematically split into a **90% Training** and **10% Validation** set (145 images for training, 17 images for verification) to ensure valid generalization boundaries.
2. **Dynamic Configuration Management:** Implemented an automated script (`create_data_yaml`) to parse local environment structures and mapping layouts into a unified, deployment-ready `data.yaml` config file.
3. **Augmentation Layer:** Applied dynamic Albumentations policies natively inside the dataloader—including structural blur filters (`Blur`, `MedianBlur`), color conversions (`ToGray`), and dynamic local contrast adaptations (`CLAHE`) to combat lighting variations.

---

## 📈 Training Progress & Fine-Tuning
The architecture leverages a lightweight backbone network to maintain a highly compact deployment footprint:
* **Model Complexity:** 225 layers, 3,007,793 parameters (Fused state).
* **Optimization Paradigm:** Fine-tuning was executed over **60 Epochs** using the **AdamW optimizer** (initial learning rate $\eta = 0.000667$, momentum $= 0.9$).
* **Regularization:** Automatic Mixed Precision (AMP) was integrated to maximize floating-point throughput while managing memory boundaries seamlessly ($2.22\text{ GB}$ peak VRAM usage).
* **Post-Processing Resolution:** Scaled to native $640\times640$ patch spatial dimensions.

### Class-by-Class Evaluation Metrics (Validation Dataset)

| Class | Instances | Precision (P) | Recall (R) | mAP50 | mAP50-95 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **All Classes** | **76** | **0.940** | **0.912** | **0.986** | **0.889** |
| MMs_peanut | 7 | 0.954 | 1.000 | 0.995 | 0.958 |
| gummy_worms | 5 | 0.983 | 1.000 | 0.995 | 0.950 |
| skittles | 2 | 0.877 | 1.000 | 0.995 | 0.945 |
| MMs_regular | 8 | 0.966 | 0.875 | 0.971 | 0.906 |
| nerds | 6 | 0.989 | 1.000 | 0.995 | 0.904 |
| milky_way | 2 | 0.591 | 1.000 | 0.995 | 0.895 |
| airheads | 17 | 0.977 | 0.941 | 0.970 | 0.882 |
| starbust | 8 | 1.000 | 0.798 | 0.982 | 0.876 |
| snickers | 7 | 1.000 | 0.900 | 0.995 | 0.875 |
| twizzlers | 7 | 1.000 | 0.803 | 0.995 | 0.801 |
| three_musketeers | 7 | 1.000 | 0.715 | 0.953 | 0.782 |

---

## 🔮 Production Inference Setup
To run localized real-time object detection predictions on your local target validation set using the absolute top weight matrix checkpoint (`best.pt`):

```bash
yolo detect predict model=runs/detect/train3/weights/best.pt source=data/validation/images save=True

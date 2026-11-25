# GTSRB Confusion-Guided Data Augmentation (CGDA)

## Completed Setup
All code is complete and ready to run:
- **E1 (Baseline):** Already trained across seeds `[42, 123, 2026]` (`Test Acc: 0.9724 ± 0.0011`, `Test Macro F1: 0.9546 ± 0.0024`).
- **E2 (`training/train_uniform.py`):** Budget-matched uniform augmentation control.
- **E3 (`training/train_static_cgda.py`):** Static CGDA using E1 validation confusion probabilities.
- **E4 (`training/train_dynamic_cgda.py`):** Dynamic CGDA updating class probabilities every 5 epochs.
- **Evaluation (`evaluation/analyze_confusion.py`, `evaluation/compare_experiments.py`):** Generates all 3-seed summaries, confusion matrices, and comparison figures.

---

## How to Run E2, E3, and E4 on Google Colab (Zero Coding Required)

1. Open [colab.research.google.com](https://colab.research.google.com) $\rightarrow$ **+ New notebook**.
2. Click **Runtime $\rightarrow$ Change runtime type $\rightarrow$ T4 GPU $\rightarrow$ Save**.
3. Copy and run each cell below in order:

### Cell 1: Mount Google Drive & Clone Repository
```python
from google.colab import drive
drive.mount('/content/drive')

!git clone [https://github.com/lizzzie917/gtsrb-cgda.git](https://github.com/lizzzie917/gtsrb-cgda.git)
%cd gtsrb-cgda

import os
os.makedirs('/content/drive/MyDrive/gtsrb-cgda/checkpoints', exist_ok=True)
os.makedirs('/content/drive/MyDrive/gtsrb-cgda/results', exist_ok=True)
!ln -s /content/drive/MyDrive/gtsrb-cgda/checkpoints checkpoints# GTSRB Confusion-Guided Data Augmentation (CGDA)

# GTSRB Confusion-Guided Data Augmentation (CGDA)

## 1. Project Overview & Current Status

This repository implements **Confusion-Guided Data Augmentation (CGDA)** on the German Traffic Sign Recognition Benchmark (GTSRB, 43 traffic sign classes) using a ResNet18 classifier.

### What Is Already Completed
* **Dataset & Track-Aware Split (`data_utils/gtsrb.py`, `results/split_seed42.json`):** Locked in an 80/20 train/validation split grouped by physical sign track so no track leaks between training and validation.
* **Experiment E1 — Baseline (`training/train_baseline.py`):** Already trained across all 3 seeds (`42`, `123`, `2026`), achieving **97.24% ± 0.11%** test accuracy and **95.46% ± 0.24%** test macro F1. Baseline metrics, confusion analysis, and figures are already saved in `results/`.
* **All Remaining Scripts Ready to Run:**
  * **E3 (`training/train_static_cgda.py`):** Static CGDA using E1 validation confusion probabilities.
  * **E2 (`training/train_uniform.py`):** Budget-matched uniform augmentation control.
  * **E4 (`training/train_dynamic_cgda.py`):** Dynamic CGDA recalculating class probabilities every 5 epochs.
  * **Evaluation (`evaluation/analyze_confusion.py`, `evaluation/compare_experiments.py`):** Aggregates 3-seed metrics, exports confusion matrices, and generates the final comparison table and plots.

### Your Role (Zero Coding Required)
You do not need to write or edit any Python code. Your task is to:
1. Run Experiments **E3**, **E2**, and **E4** on Google Colab using the copy-paste cells below.
2. Run the evaluation scripts in Colab to generate all metrics, confusion matrices, and comparison charts.
3. Download the generated `results_all.zip` file and upload the new results back to this GitHub repository.

---

## 2. Phase 1: Accept the GitHub Collaborator Invite

1. Open your email inbox (the email linked to your GitHub account) and look for an invitation email from GitHub for **`lizzzie917/gtsrb-cgda`**.
   * *Alternative:* Sign in to GitHub in your browser and visit `https://github.com/lizzzie917/gtsrb-cgda/invitations`.
2. Click **Accept invitation**.
3. Confirm you can view the repository homepage at `https://github.com/lizzzie917/gtsrb-cgda`.

---

## 3. Phase 2: Open Google Colab & Enable GPU

1. Open a web browser (Chrome or Edge recommended) and go to `https://colab.research.google.com`.
2. Sign in with your Google account (make sure you have at least 2 GB of free space on Google Drive for model checkpoints and datasets).
3. In the pop-up window, click **+ New notebook** at the bottom (or click **File -> New notebook in Drive** in the top-left menu).
4. Optional: Click the notebook title at the top left (e.g., `Untitled0.ipynb`) and rename it to `GTSRB_CGDA_Experiments.ipynb`.
5. Turn on the GPU:
   * Click **Runtime** in the top menu bar.
   * Click **Change runtime type**.
   * Under **Hardware accelerator**, select **T4 GPU**.
   * Click **Save**.
6. Wait a few seconds until the top-right corner displays a green checkmark next to **RAM / Disk** (this confirms your GPU machine is connected).

---

## 4. Phase 3: Run Experiments E3, E2, and E4 in Google Colab

### How to Use Colab Cells
* To add a new code block, hover near the bottom center of the current cell and click **+ Code** (or click **+ Code** in the top-left toolbar).
* Paste the exact code from one cell below into the Colab code box.
* Click the **Play button** (circle with a triangle) on the left side of the cell, or press **`Shift + Enter`**.
* **Wait for the cell to finish completely** (the spinning circle around the Play button will stop and turn into a green checkmark) before running the next cell.

---

### Cell 1: Mount Google Drive & Clone the Repository
> **What this does:** Connects your Google Drive so results are backed up automatically, downloads the project code from GitHub, and downloads/prepares folders. When prompted by a pop-up window, click **Connect to Google Drive**, choose your Google account, and click **Allow**.

```python
from google.colab import drive
drive.mount('/content/drive')

# Clone the repository using a plain URL
!git clone https://github.com/lizzzie917/gtsrb-cgda.git

# Enter the repository
%cd /content/gtsrb-cgda

# Create backup directories in Google Drive
import os

os.makedirs('/content/drive/MyDrive/gtsrb-cgda/checkpoints', exist_ok=True)
os.makedirs('/content/drive/MyDrive/gtsrb-cgda/results', exist_ok=True)

# Link checkpoints to Google Drive
if os.path.islink('checkpoints') or os.path.exists('checkpoints'):
    os.remove('checkpoints')

os.symlink(
    '/content/drive/MyDrive/gtsrb-cgda/checkpoints',
    'checkpoints'
)

# Verify repository location
print("Current directory:", os.getcwd())
print("Training folder exists:", os.path.isdir('training'))
print("Results folder exists:", os.path.isdir('results'))
```

---
### Cell 2, 3, 4: Install dependencies
> **Before training, run this in a separate cell:**

```
!pip install -r requirements.txt
```

> **verify that Colab recognizes your GPU:**

```
import torch

print("PyTorch version:", torch.__version__)
print("GPU available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

> **verify that the training scripts are accessible:**

```
!ls training/
!ls results/metrics/
!ls results/analysis/
```

---

### Cell 5, 6: Run Experiment E3 (Static CGDA — 3 Seeds)
> **What this does:** Downloads the GTSRB dataset on the first seed, trains Static CGDA for 25 epochs across seeds `42`, `123`, and `2026`, and backs up `results/` to your Google Drive. (Run time: 35 minutes).*

```python
!python -m training.train_static_cgda --seed 42
!python -m training.train_static_cgda --seed 123
!python -m training.train_static_cgda --seed 2026
!cp -r results/* /content/drive/MyDrive/gtsrb-cgda/results/
```

> **One important precaution:** You can run all three seeds in one cell but if you want, you can also run them in separate cells. Just don't run them all in separate Colab cells simultaneously. Run them SEQUENTIALLY so that GPU memory and experiment outputs are managed properly. 

> **verify its output:** 

```python
 !ls results/metrics/E3*
```

---

### Cell 6: Run Experiment E2 (Budget-Matched Uniform Augmentation — 3 Seeds)
> **What this does:** Trains the uniform augmentation control across seeds `42`, `123`, and `2026`, then backs up `results/` to your Google Drive. *(Estimated time: 10 to 15 minutes).*

```python
!python -m training.train_uniform --seed 42
!python -m training.train_uniform --seed 123
!python -m training.train_uniform --seed 2026
!cp -r results/* /content/drive/MyDrive/gtsrb-cgda/results/
```
> **verify its output:** 

```python
 !ls results/metrics/E2*
```
---

### Cell 7: Run Experiment E4 (Dynamic CGDA — 3 Seeds)
> **What this does:** Trains Dynamic CGDA (updating per-class augmentation probabilities every 5 epochs) across seeds `42`, `123`, and `2026`, then backs up `results/` to your Google Drive. *(Estimated time: 10 to 15 minutes).*

```python
!python -m training.train_dynamic_cgda --seed 42
!python -m training.train_dynamic_cgda --seed 123
!python -m training.train_dynamic_cgda --seed 2026
!cp -r results/* /content/drive/MyDrive/gtsrb-cgda/results/
```

> **Tip if Colab disconnects mid-way:** Keep the Colab browser tab open while training runs. If Colab ever disconnects or restarts, re-run **Cell 1**, restore any already-finished results from Drive by running `!cp -r /content/drive/MyDrive/gtsrb-cgda/results/* results/`, and then continue from whichever experiment cell did not finish.

> **verify its output:** 

```python
 !ls results/metrics/E4*
```
---

## 5. Phase 4: Generate Confusion Analyses, Charts & Download Results

### Cell 5: Generate All Summary Tables, Confusion Matrices & Comparison Plots
> **What this does:** Evaluates all 3 seeds for `E2_uniform`, `E3_static_cgda`, and `E4_dynamic_cgda`, creates the side-by-side comparison table and figure across `E1` through `E4`, and backs everything up to Google Drive. *(Estimated time: under 15 seconds).*

```python
!python -m evaluation.analyze_confusion --exp E2_uniform
!python -m evaluation.analyze_confusion --exp E3_static_cgda
!python -m evaluation.analyze_confusion --exp E4_dynamic_cgda
!python -m evaluation.compare_experiments
!cp -r results/* /content/drive/MyDrive/gtsrb-cgda/results/
```

---

### Cell 6: Zip & Download `results_all.zip` to Your Computer
> **What this does:** Packages the entire `results/` folder into `results_all.zip` and triggers a browser download to your computer's **Downloads** folder.

```python
!zip -r /content/results_all.zip results/
from google.colab import files
files.download('/content/results_all.zip')
```

*(If your browser blocks the automatic download, click the **Folder icon** on the far-left sidebar of Colab, click the `..` parent folder icon to go to `/content/`, find `results_all.zip`, click the three dots next to it, and click **Download**).*

---

### If you don't want to Download `results_all.zip` to Your Computer, you can just Commit Changes directly to Github
> **Follow Phase 5.2 Instructions**.


## 6. Phase 5.1: Upload the Results Back to GitHub (Local Computer Ver)

### Step 1: Clone the Repository to Your Computer
1. Open a terminal on your computer:
   * **Windows:** Press the **Windows key**, type `PowerShell` or `Git Bash`, and press **Enter**.
   * **Mac:** Press **`Cmd + Space`**, type `Terminal`, and press **Enter**.
2. Navigate to where you want to store the project (for example, your `Documents` folder) and clone the repository:
   ```bash
   cd ~/Documents
   git clone [https://github.com/lizzzie917/gtsrb-cgda.git](https://github.com/lizzzie917/gtsrb-cgda.git)
   cd gtsrb-cgda
   git checkout -b feature/cgda-results
   ```

### Step 2: Extract `results_all.zip` into Your Local `gtsrb-cgda/results/` Folder
1. Open your computer's file manager (**File Explorer** on Windows or **Finder** on Mac) and open your **Downloads** folder.
2. Right-click **`results_all.zip`** and extract/unzip it:
   * **Windows:** Right-click `results_all.zip` -> **Extract All...** -> **Extract**.
   * **Mac:** Double-click `results_all.zip`.
3. Open the extracted folder until you see the **`results`** folder containing three subfolders: `analysis`, `figures`, and `metrics`.
4. Copy those three folders (`analysis`, `figures`, and `metrics`) and paste them into your cloned repository's **`Documents/gtsrb-cgda/results/`** folder.
   * When asked if you want to merge or replace existing files, click **Replace / Merge**.
   * **Important:** Do **not** delete `results/split_seed42.json` or any `E1_baseline` files already inside `results/`.

### Step 3: Push Your Results Branch & Open a Pull Request
1. Return to your terminal inside the `gtsrb-cgda` folder and run:
   ```bash
   git status
   git add results/
   git commit -m "Add E2, E3, and E4 metrics, confusion analyses, and figures"
   git push -u origin feature/cgda-results
   ```
2. Open `https://github.com/lizzzie917/gtsrb-cgda` in your web browser.
3. Click the green **Compare & pull request** button that appears near the top of the page.
4. Click **Create pull request**, then click **Merge pull request** -> **Confirm merge** to merge your results into `main`.



## 7. Phase 5.2: Upload the Results Directly to GitHub (Collab Ver)
### Step 1: Configure your Git identity
1. Run this once:
   ```bash
   !git config --global user.name "Your GitHub Name"
   !git config --global user.email "your-github-email@example.com"
   ```
   Use the name and email associated with your GitHub account.

### Step 2: Check which files have changed
```python
!git status
```
You should see your newly generated experiment results.

### Step 3: Stage your results
1. If you only want to commit experiment outputs:
   ```bash
   !git add results/
   ```
2. If you only want to commit experiment outputs:
   ```bash
   !git add checkpoints/
   ```

### Step 4: Commit the changes
```python
!git commit -m "Add E2 E3 and E4 experimental results"
```

### Step 5: Push to Github
1. First, check your current branch:
   ```bash
   !git branch --show-current
   ```
2. Then push:
   ```bash
   !git push origin main
   ```
   If your branch is not main, replace it with the branch name shown.

### Authentication
If GitHub requests authentication, use a Personal Access Token (PAT) or GitHub CLI authentication.

Do not put your PAT directly into a notebook cell because it can remain visible in notebook history.

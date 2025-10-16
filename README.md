# Speech-Classifier-DTW

This project was completed as part of the **Deep Learning for Speech Signals** course (Technion, Jan 2025).  
It includes: (1) DTW vs. Euclidean 1-NN digit recognition with MFCC features, and (3) speaker/command classification with x-vectors.

---

## 📑 Project Overview

### Part 1 — DTW vs. Euclidean (1-NN on MFCCs)
- Implement a 1-NN classifier for digits **1..5** using **Euclidean** distance and **DTW** (you must implement DTW yourself). The MFCC feature shape is **(20, 32)** per 1-second audio. :contentReference[oaicite:0]{index=0}  
- Dataset: **5 labeled train examples per class**; **253** unlabeled test examples. Predictions go to `output.txt` in the format:  
  `sampleX.wav - <euclidean_pred> - <dtw_pred>`. :contentReference[oaicite:1]{index=1}

### Part 3 — X-vectors (SpeechBrain) for Classification
- Use pretrained **x-vectors** (SpeechBrain) as embeddings; visualize (t-SNE) and evaluate several classifiers with CV. :contentReference[oaicite:2]{index=2}  
- Our report includes results and discussion for both **speaker** and **command** classification. :contentReference[oaicite:3]{index=3}

---

## 📂 Project Structure
```
DLSpeech_Ex4/
├─ docs/
│ ├─ DLSpeech__Ex4_046747.pdf # Assignment instructions
│ ├─ ex_4_report_313551186_205853633.pdf # Submitted report (parts 2 & 3)
├─ data/
│ ├─ train_data/ # training WAVs (not stored in repo)
│ └─ test_files/ # test WAVs (not stored in repo)
├─ src/
│ ├─ ex_4_part1.py # MFCC + DTW/Euclidean 1-NN
│ └─ ex_4_part3.ipynb # X-vectors experiments (Colab)
├─ tools/
│ └─ sanity_check.py # verifies output.txt format
├─ results/
│ └─ output.txt # our predictions file (253 lines)
├─ .gitignore
├─ LICENSE
└─ README.md
```

---

## 📊 Data

The train/test WAV files are large, so they aren’t stored directly in the repo.  
Place them locally under:
```
data/train_data/
data/test_files/
```
(Optionally host them on Drive/Releases and link here.)

---

## ⚙️ Requirements
- Python 3.9+  
- `librosa`, `numpy`, `natsort` (for Part 1)  
- `speechbrain`, `sklearn`, `matplotlib` (for Part 3, in the notebook)

Install (example):
```bash
pip install numpy librosa natsort
pip install speechbrain scikit-learn matplotlib
```
▶️ How to Run (Part 1)

1. Generate predictions:
```bash
python src/ex_4_part1.py
```
(Implementation computes MFCCs with librosa.feature.mfcc(..., n_mfcc=20)[:,:32], runs Euclidean and custom DTW distances, and writes output.txt.) 

2. Validate format:

```bash
python tools/sanity_check.py
```
The checker expects 253 lines and specific sample predictions (e.g., sample1.wav -> [4,4], sample2.wav -> [1,1], sample3.wav -> [3,3]). On success it prints “output.txt is in the correct format”.
3. Example output.txt (ours) is included under results/.

📓 Part 3 (X-vectors)

Run src/ex_4_part3.ipynb (Colab recommended) to:

Extract x-vector embeddings (SpeechBrain) and visualize (t-SNE).

Evaluate multiple classifiers with CV and report accuracy/std (see docs/ex_4_report_...pdf).

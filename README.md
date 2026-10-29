# Hidden Markov Models for Human Activity Recognition


Assignment: modelling Standing, Walking, Jumping and Still from smartphone accelerometer & gyroscope data using a continuous Gaussian HMM implemented **from scratch** with NumPy.

---

## Repository Structure

```
HMM-Human-Activity-Recognition/
├── notebooks/
│   └── HMM_Activity_Recognition.ipynb   # complete pipeline (Colab / local)
├── figures/                             # all plots used in the report
│   ├── sample_signals.png
│   ├── transition_matrix.png
│   ├── confusion_matrix.png
│   ├── decoded_sequences.png
│   └── bw_convergence.png
├── requirements.txt
└── README.md
```

The original Sensor Logger recordings stay on Google Drive (or a local folder) and are loaded live by the notebook.

---

## Quick Start – Google Colab (recommended)

1. Keep your original Sensor Logger zip files in a Drive folder  
   (example path used in the notebook):
   ```
   MyDrive/MLTechII-Formative2-sample-files
   ```
   Each file should look like `jumping_01_csv-2026-10-29_11-16-04.zip`, `standing_03_...`, etc.

2. Upload this repository (or just the notebook) to Drive and open  
   `notebooks/HMM_Activity_Recognition.ipynb` in Google Colab.

3. In the first code cell set the correct path to your recordings:
   ```python
   RAW_DATA_DIR = "/content/drive/MyDrive/MLTechII-Formative2-sample-files"
   ```

4. Runtime → Run all.

The notebook will:
- mount Google Drive,
- open every original Sensor Logger zip,
- merge Total Acceleration + Gyroscope streams,
- extract time- and frequency-domain features,
- train a Gaussian HMM (Baum-Welch + Viterbi) entirely from scratch,
- evaluate on held-out recordings,
- produce every figure required by the assignment.

---

## Running outside Colab (VS Code, local Jupyter, Kaggle, etc.)

The notebook was written for Google Colab, so two small changes are needed:

### 1. Disable / remove the Drive-mount cell

At the top of the notebook you will find a cell that starts with:

```python
from google.colab import drive
drive.mount('/content/drive')
...
```

**Either delete that cell or comment it out.**  
It will raise an error on any non-Colab environment.

### 2. Point `RAW_DATA_DIR` to a local folder

Replace the Colab path with the location of your original Sensor Logger zip files on your machine, for example:

```python
# Windows example
RAW_DATA_DIR = r"C:\Users\Francis\Documents\MLTechII-Formative2-sample-files"

# macOS / Linux example
RAW_DATA_DIR = "/home/francis/data/MLTechII-Formative2-sample-files"
```

Make sure the folder contains the original `.zip` files produced by Sensor Logger (the ones named `activity_XX_csv-....zip`).

### 3. Install dependencies (if needed)

```bash
pip install -r requirements.txt
```

(`numpy`, `pandas`, `matplotlib`, `seaborn` are sufficient.)

After these two changes the rest of the notebook runs unchanged on VS Code, Jupyter Lab, or any standard Python environment.

---

## Data Collection Summary

| Item                      | Detail                                         |
|---------------------------|------------------------------------------------|
| Device                    | itel A665L (Android)                           |
| App                       | Sensor Logger (Logger Labs Ltd)                |
| Sensors                   | Total Acceleration (x,y,z) + Gyroscope (x,y,z) |
| Sampling rate             | **100 Hz** (sampleRateMs = 10)                 |
| Activities                | Standing, Walking, Jumping, Still              |
| Files per activity        | 20                                             |
| Duration per file         | 5–10 seconds                                   |
| Total duration / activity | > 1 min 30 s                                   |

---

## Model Summary

| Item                  | Value                                              |
|-----------------------|----------------------------------------------------|
| Hidden states         | 4 (standing, walking, jumping, still)              |
| Features per window   | 8 (4 time-domain + 2 gyro + 2 frequency)           |
| Window / hop          | 1.0 s / 0.5 s                                      |
| Emission model        | Diagonal Gaussian                                  |
| Training algorithm    | Baum-Welch (log-likelihood convergence ε = 1e-3)   |
| Decoding algorithm    | Viterbi                                            |
| Test-set accuracy     | **≈ 95 %**                                         |

---

## Video Proof of Data Collection

Four short (~5 s) videos showing the author performing each activity while holding the phone (full body + face visible) must be linked below:

- Standing: *[insert link]*
- Walking: *[insert link]*
- Jumping: *[insert link]*
- Still: *[insert link]*

---

## Requirements

```
numpy
pandas
matplotlib
seaborn
```

(Already available in Google Colab; listed for local / VS Code runs.)

---

## Licence

For educational use only.

## Author:

Francis Mutabazi — an aspiring ML engineer

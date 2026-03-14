# Kernel PCA — Nonlinear Dimensionality Reduction

Demonstrates **Kernel Principal Component Analysis (Kernel PCA)** on a social-network ads dataset, using both Python and R. The pipeline reduces features with an RBF kernel, trains a Logistic Regression classifier, and visualises decision boundaries for training and test sets.

---

## Methodology

1. Load the `Social_Network_Ads.csv` dataset (Age, Estimated Salary → Purchased).
2. Split into 75 % training / 25 % test.
3. Standardise features (zero mean, unit variance).
4. Apply **Kernel PCA** with an RBF kernel, projecting down to 2 components.
5. Fit **Logistic Regression** on the transformed training set.
6. Evaluate with a confusion matrix and plot decision regions for both sets.

---

## Tech Stack

| Layer | Tool |
|---|---|
| 🐍 Language (primary) | Python 3 |
| 📊 Language (secondary) | R |
| 🔬 ML Framework | scikit-learn |
| 🧮 Numerics | NumPy |
| 📈 Visualisation | Matplotlib |
| 🗂️ Data Handling | pandas |
| 📦 R Packages | `caTools`, `kernlab`, `ElemStatLearn` |

---

## Dependencies

### Python

```
numpy
pandas
matplotlib
scikit-learn
```

Install with:

```bash
pip install numpy pandas matplotlib scikit-learn
```

### R

```r
install.packages(c("caTools", "kernlab", "ElemStatLearn"))
```

---

## How to Run

### Python

```bash
cd Kernel-PCA
python kernel_pca.py
```

### R

```bash
Rscript kernel_pca.R
```

> Both scripts expect `Social_Network_Ads.csv` in the working directory.

---

## Known Issues

- The R script calls `install.packages('ElemStatLearn')` at runtime, which may fail in non-interactive sessions. Install the package beforehand or comment out that line.
- Visualisation step generates a dense meshgrid (`step = 0.01`) and can be slow on large displays or low-memory machines. Increase the step size if needed.
- The `ElemStatLearn` R package has been archived from CRAN; you may need to install it from a mirror or GitHub archive.

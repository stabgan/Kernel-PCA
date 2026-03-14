# Kernel PCA — Non-Linear Dimensionality Reduction

Applies **Kernel PCA** (RBF kernel) to reduce a 2-D social-network ads dataset, then classifies purchase intent with **Logistic Regression**. Implementations in both Python and R.

## What It Does

1. Loads the *Social Network Ads* dataset (age, estimated salary → purchased).
2. Scales features with `StandardScaler` (Python) / `scale()` (R).
3. Projects data into a 2-component kernel PCA space using an RBF kernel.
4. Trains a logistic regression classifier on the transformed features.
5. Visualises the non-linear decision boundary for training and test sets.

## Dataset

| File | Rows | Features | Target |
|---|---|---|---|
| `Social_Network_Ads.csv` | 400 | Age, Estimated Salary | Purchased (0/1) |

## 🛠 Tech Stack

| | Tool | Purpose |
|---|---|---|
| 🐍 | Python 3 | Main implementation |
| 📊 | scikit-learn | KernelPCA, LogisticRegression, StandardScaler |
| 📈 | matplotlib | Decision-boundary visualisation |
| 🔢 | NumPy / pandas | Data handling |
| 📐 | R | Alternative implementation |
| 📦 | kernlab (R) | Kernel PCA via `kpca()` |
| 📦 | caTools (R) | Train/test split |

## Getting Started

### Python

```bash
pip install numpy pandas matplotlib scikit-learn
python kernel_pca.py
```

### R

```r
# install.packages(c("caTools", "kernlab"))
source("kernel_pca.R")
```

## ⚠️ Known Issues

- The R version previously depended on `ElemStatLearn`, which has been removed from CRAN. That dependency has been dropped — all plotting uses base R.
- Axis labels now correctly read "PC 1" / "PC 2" instead of the raw feature names, since Kernel PCA transforms the original feature space.

## License

See [LICENSE](LICENSE).

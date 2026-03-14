# Kernel PCA — dimensionality reduction + logistic regression on Social Network Ads

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Resolve dataset path relative to this script so it works from any cwd
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CSV_PATH = os.path.join(_SCRIPT_DIR, "Social_Network_Ads.csv")


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    dataset = pd.read_csv(_CSV_PATH)
    X = dataset.iloc[:, [2, 3]].values
    y = dataset.iloc[:, 4].values

    # ------------------------------------------------------------------
    # 2. Train / test split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    # ------------------------------------------------------------------
    # 3. Feature scaling
    # ------------------------------------------------------------------
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # ------------------------------------------------------------------
    # 4. Kernel PCA (RBF kernel, 2 components)
    # ------------------------------------------------------------------
    kpca = KernelPCA(n_components=2, kernel="rbf")
    X_train = kpca.fit_transform(X_train)
    X_test = kpca.transform(X_test)

    # ------------------------------------------------------------------
    # 5. Logistic Regression
    # ------------------------------------------------------------------
    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 6. Predictions & confusion matrix
    # ------------------------------------------------------------------
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:\n", cm)

    # ------------------------------------------------------------------
    # 7. Visualise results
    # ------------------------------------------------------------------
    _plot_decision_boundary(
        classifier, X_train, y_train, title="Logistic Regression (Training set)"
    )
    _plot_decision_boundary(
        classifier, X_test, y_test, title="Logistic Regression (Test set)"
    )


def _plot_decision_boundary(
    classifier, X_set: np.ndarray, y_set: np.ndarray, title: str
) -> None:
    """Plot the decision boundary with a filled contour and scatter overlay."""
    colors = ("red", "green")
    cmap = ListedColormap(colors)

    X1, X2 = np.meshgrid(
        np.arange(X_set[:, 0].min() - 1, X_set[:, 0].max() + 1, 0.01),
        np.arange(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1, 0.01),
    )
    Z = classifier.predict(np.c_[X1.ravel(), X2.ravel()]).reshape(X1.shape)

    plt.contourf(X1, X2, Z, alpha=0.75, cmap=cmap)
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())

    for idx, label in enumerate(np.unique(y_set)):
        # FIX: pass a plain color string instead of calling ListedColormap()(i),
        # which returns an RGBA tuple and can break across matplotlib versions.
        plt.scatter(
            X_set[y_set == label, 0],
            X_set[y_set == label, 1],
            c=colors[idx],
            label=label,
        )

    plt.title(title)
    # After Kernel PCA the axes are principal components, not the raw features.
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve  # noqa: E402

plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
sns.set()


def save_fig(fig_path, tight_layout=True, resolution=150):
    '''
    Saves the current matplotlib figure and closes it.
    :param fig_path: output path including extension
    :param tight_layout: call plt.tight_layout() first
    :param resolution: dpi
    :return: None
    '''
    os.makedirs(os.path.dirname(str(fig_path)) or ".", exist_ok=True)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(fig_path, dpi=resolution)
    plt.close()


def plot_roc_curve(y_true, y_score, title, label=None):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], "k--")
    plt.axis([0, 1, 0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    if label:
        plt.legend(loc="lower right")


def plot_precision_recall_vs_threshold(y_true, y_score, title, threshold=None):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    if threshold is not None:
        plt.axvline(threshold, color="k", linestyle=":", label=f"threshold = {threshold:.3f}")
    plt.xlabel("Threshold")
    plt.ylim([0, 1])
    plt.legend(loc="center right")
    plt.title(title)


def plot_confusion_matrix(y_true, y_pred, title):
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, square=True, annot=True, annot_kws={"size": 16}, fmt="g", cbar=False)
    plt.xlabel("predicted")
    plt.ylabel("ground truth")
    plt.title(title)


def plot_calibration(decile_table, title):
    plt.figure(figsize=(7, 6))
    upper = max(decile_table["mean_predicted"].max(), decile_table["observed_rate"].max()) * 1.05
    plt.plot([0, upper], [0, upper], "k--")
    plt.plot(decile_table["mean_predicted"], decile_table["observed_rate"], "o-")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed rate")
    plt.title(title)


def plot_shap_summary(contributions, X, title, max_display=20):
    '''
    SHAP summary plot from XGBoost's native contributions (pred_contribs=True), which
    avoids version coupling between the shap and xgboost packages.
    :param contributions: array (n_rows, n_features + 1); the last column is the bias
    :param X: feature DataFrame used for the contributions
    '''
    import shap

    shap.summary_plot(np.asarray(contributions)[:, :-1], X, max_display=max_display, show=False)
    plt.title(title)

import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(cm, labels, show=True, save_path=None, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(labels))); ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right"); ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.tight_layout()
    if save_path: plt.savefig(save_path, bbox_inches="tight")
    if show: plt.show()
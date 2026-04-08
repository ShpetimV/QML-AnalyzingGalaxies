import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from src.param_config import VisualConfig


class SDSSMetricTracker:
    def __init__(self, results_dir="./results"):

        self.vis_config = VisualConfig()

        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        # Use the dark astronomy theme from the original code
        self.bg_color = self.vis_config.bg_color
        self.text_color = self.vis_config.text_color
        self.grid_color = self.vis_config.grid_color

    def plot_history(self, history):
        """Plots training and validation curves from the trainer's history dict."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor(self.bg_color)

        epochs = range(1, len(history['train_loss']) + 1)

        for ax, metric, title in zip([ax1, ax2], ['loss', 'acc'], ['Loss', 'Accuracy']):
            ax.set_facecolor(self.bg_color)
            ax.plot(epochs, history[f'train_{metric}'], color=self.vis_config.accent_color, label='Train', linewidth=2)
            ax.plot(epochs, history[f'val_{metric}'], color=self.vis_config.warning_color, label='Val', linewidth=2)
            ax.set_title(f'Training {title}', color='white', fontsize=14)
            ax.set_xlabel('Epoch', color='#888888')
            ax.tick_params(colors='#888888')
            ax.legend()
            for spine in ax.spines.values():
                spine.set_edgecolor(self.grid_color)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "training_history.png"), facecolor=self.bg_color)
        plt.close()

    def plot_confusion_matrix(self, y_true, y_pred, class_names, split_name='test'):
        """Generates a row-normalized confusion matrix to identify hard-to-classify objects."""
        n = len(class_names)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))

        # Normalize by row (Recall) to see how many of each true class were caught
        cm_norm = np.divide(cm.astype(float), cm.sum(axis=1, keepdims=True),
                            where=cm.sum(axis=1, keepdims=True) != 0)

        fig_size = max(10, int(n * 0.7))
        fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.8))
        fig.patch.set_facecolor(self.bg_color)
        ax.set_facecolor(self.bg_color)

        im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(colors='#888888')

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(class_names, rotation=45, ha='right', color=self.text_color)
        ax.set_yticklabels(class_names, color=self.text_color)

        # Annotate with percentages
        for i, j in itertools.product(range(n), range(n)):
            val = cm_norm[i, j]
            if val > 0.01:
                ax.text(j, i, f'{val:.2f}', ha="center", va="center",
                        color="white" if val > 0.5 else "#aaaaaa", fontsize=8)

        ax.set_ylabel('True Label', color=self.text_color, fontsize=12)
        ax.set_xlabel('Predicted Label', color=self.text_color, fontsize=12)
        ax.set_title(f'Confusion Matrix - {split_name.upper()}', color='white', pad=20)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f"cm_{split_name}.png"), facecolor=self.bg_color)
        plt.close()

    def plot_per_class_accuracy(self, y_true, y_pred, class_names):
        """Bar chart of accuracy per subclass, sorted from easiest to hardest."""
        n = len(class_names)
        accs = []
        for i in range(n):
            mask = (y_true == i)
            acc = (y_pred[mask] == i).mean() if mask.sum() > 0 else 0
            accs.append((class_names[i], acc))

        accs.sort(key=lambda x: x[1], reverse=True)
        names, values = zip(*accs)

        plt.figure(figsize=(max(12, int(n * 0.5)), 6))
        plt.gcf().set_facecolor(self.bg_color)
        ax = plt.gca()
        ax.set_facecolor(self.bg_color)

        colors = [self.vis_config.accent_color if v >= 0.8 else self.vis_config.warning_color if v >= 0.5 else self.vis_config.error_color for v in values]
        plt.bar(names, values, color=colors, edgecolor=self.grid_color)

        plt.axhline(y=0.8, color=self.vis_config.accent_color, linestyle='--', alpha=0.3)
        plt.title("Per-Class Accuracy (Hardest Classes on the Right)", color='white')
        plt.xticks(rotation=45, ha='right', color=self.text_color)
        plt.yticks(color=self.text_color)
        plt.ylabel("Accuracy / Recall", color=self.text_color)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "per_class_accuracy.png"), facecolor=self.bg_color)
        plt.close()
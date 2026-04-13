import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from src.param_config import VisualConfig
import pandas as pd


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

    def analyze_confusion_matrix(self, y_true, y_pred, class_names, split_name='test'):
        """
        Saves the CM to a CSV and generates a smart text report identifying QML targets.
        """

        n = len(class_names)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))

        # 1. SAVE RAW AND NORMALIZED DATA TO CSV
        df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
        df_cm.to_csv(os.path.join(self.results_dir, f"cm_raw_{split_name}.csv"))

        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
        df_cm_norm = pd.DataFrame(cm_norm, index=class_names, columns=class_names)
        df_cm_norm.to_csv(os.path.join(self.results_dir, f"cm_norm_{split_name}.csv"))

        # 2. SMART ANALYSIS LOGIC
        report_lines = []
        report_lines.append(f"=== QML TARGET ANALYSIS REPORT ({split_name.upper()} SET) ===\n")
        report_lines.append(f"Total Classes: {n}")
        report_lines.append(f"Total Samples: {cm.sum()}\n")

        # A. Find "The Hardest Classes" (Lowest Recall)
        recalls = np.diag(cm_norm)
        hardest_idx = np.argsort(recalls)
        report_lines.append("--- 1. THE HARDEST CLASSES (LOWEST ACCURACY/RECALL) ---")
        report_lines.append("These are classes the model consistently fails to identify.")
        for idx in hardest_idx[:10]:  # Top 10 worst
            if row_sums[idx][0] > 0:  # Only count if there were actual samples
                report_lines.append(
                    f" - {class_names[idx]:<20}: {recalls[idx] * 100:.1f}% accuracy ({row_sums[idx][0]} samples)")

        # B. Find "Mirrors" (Highest Bidirectional Confusion)
        # A <-> B (Both get confused for each other)
        report_lines.append("\n--- 2. THE MIRRORS (SYMMETRIC CONFUSION PAIRS) ---")
        report_lines.append("Prime QML Targets: The CNN sees these pairs as overlapping in feature space.")
        sym_confusions = []
        for i in range(n):
            for j in range(i + 1, n):
                if row_sums[i][0] > 0 and row_sums[j][0] > 0:
                    sym_val = cm_norm[i, j] + cm_norm[j, i]
                    if sym_val > 0.10:  # Only care if combined confusion > 10%
                        sym_confusions.append((sym_val, class_names[i], class_names[j], cm_norm[i, j], cm_norm[j, i]))

        sym_confusions.sort(key=lambda x: x[0], reverse=True)
        for val, c1, c2, c1_to_c2, c2_to_c1 in sym_confusions[:10]:
            report_lines.append(f" - {c1} <---> {c2} | (Total overlap: {val * 100:.1f}%)")
            report_lines.append(f"      {c1} predicted as {c2}: {c1_to_c2 * 100:.1f}%")
            report_lines.append(f"      {c2} predicted as {c1}: {c2_to_c1 * 100:.1f}%")

        # C. Find "Black Holes" (Highest False Positives / Sinks)
        report_lines.append("\n--- 3. THE BLACK HOLES (FALSE POSITIVE SINKS) ---")
        report_lines.append("Classes that the model defaults to when it is uncertain.")
        false_positives = cm.sum(axis=0) - np.diag(cm)
        sink_idx = np.argsort(false_positives)[::-1]
        for idx in sink_idx[:10]:
            report_lines.append(f" - {class_names[idx]:<20}: Absorbed {false_positives[idx]} false predictions.")

        # D. Detect Complex Clusters (Triplets/Groups)
        report_lines.append("\n--- 4. PROPOSED QML SUB-PROBLEMS (CLUSTERS) ---")
        report_lines.append("If you want to train a Quantum Circuit on 3-4 classes, pick one of these groups:")

        visited = set()
        clusters_found = 0
        for i in hardest_idx:
            if i in visited or row_sums[i][0] == 0: continue

            # Find classes where > 15% of class 'i' goes
            confused_with = np.nonzero(cm_norm[i] > 0.15)[0]
            cluster = {i}
            for j in confused_with:
                if j != i: cluster.add(j)

            if len(cluster) >= 3:
                clusters_found += 1
                cluster_names = [class_names[c] for c in cluster]
                report_lines.append(f" Group {clusters_found}: {', '.join(cluster_names)}")
                visited.update(cluster)

            if clusters_found >= 5: break  # Limit to top 5 clusters

        # 3. WRITE TO TEXT FILE
        report_path = os.path.join(self.results_dir, f"qml_target_analysis_{split_name}.txt")
        with open(report_path, "w") as f:
            f.write("\n".join(report_lines))

        print(f"✅ QML Analysis Report saved to {report_path}")
        print(f"✅ Raw Data matrices saved to CSVs in {self.results_dir}")
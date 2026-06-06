import torch
import torch.nn as nn
from sklearn.svm import SVC

class SmallDenseNet(nn.Module):
    """
    A Featherweight Multi-Layer Perceptron (MLP) for few-shot benchmarking.
    Takes the 8D PCA features and maps them to the num_classes.
    """
    def __init__(self, input_dim=8, hidden_dim=16, num_classes=2, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def get_classical_svm(c_param=1.0, kernel='rbf'):
    """
    Returns a standard Scikit-Learn SVM.
    Probability=True is required so we can plot ROC/PR curves later.
    """
    return SVC(kernel=kernel, C=c_param, probability=True, random_state=42, gamma='scale')
# core/visualizations.py
import os
import uuid
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

def plot_confusion_matrix(y_true, y_pred, save_dir='media'):
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    filename = f"conf_matrix_{uuid.uuid4().hex}.png"
    path = os.path.join(save_dir, filename)
    plt.savefig(path)
    plt.close()
    return filename

def plot_roc_curve(y_true, y_proba, save_dir='media'):
    os.makedirs(save_dir, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    filename = f"roc_curve_{uuid.uuid4().hex}.png"
    path = os.path.join(save_dir, filename)
    plt.savefig(path)
    plt.close()
    return filename

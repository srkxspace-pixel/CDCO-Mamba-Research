import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score
import torch
from sklearn.preprocessing import label_binarize

def calculate_metrics(pred, label):
    pred_scores = pred
    pred = np.argmax(pred, axis=-1)
    label = np.argmax(label, axis=-1)
    conf_matrix = confusion_matrix(label, pred)
    accuracy = accuracy_score(label, pred)
    specificity = []
    sensitivity = []
    for i in range(conf_matrix.shape[0]):
        tn = np.sum(np.delete(np.delete(conf_matrix, i, axis=0), i, axis=1)) # True Negatives
        fp = np.sum(np.delete(conf_matrix[i, :], i))                        # False Positives
        specificity.append(tn / (tn + fp))
        tp = conf_matrix[i, i]                                             # True Positives
        fn = np.sum(conf_matrix[:, i]) - tp                                # False Negatives
        sensitivity.append(tp / (tp + fn))
    avg_specificity = np.mean(specificity)
    avg_sensitivity = np.mean(sensitivity)
    f1 = f1_score(label, pred, average='macro')
    classes = list(range(np.shape(pred_scores)[-1]))  # upstream hard-coded [0, 1, 2, 3]
    label_bin = label_binarize(label, classes=classes)
    pred_bin = label_binarize(pred, classes=classes)
    auc = roc_auc_score(label_bin, pred_bin, average='macro', multi_class='ovo')
    return accuracy, avg_specificity, avg_sensitivity, f1, auc

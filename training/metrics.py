import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score
import torch

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
    # AUC from predicted probabilities (upstream binarised the argmax, i.e. AUC of hard labels)
    probs = torch.softmax(torch.tensor(np.asarray(pred_scores, dtype=np.float64)), -1).numpy()
    if probs.shape[1] == 2:
        auc = roc_auc_score(label, probs[:, 1])
    else:
        auc = roc_auc_score(label, probs, average='macro', multi_class='ovr', labels=list(range(probs.shape[1])))
    return accuracy, avg_specificity, avg_sensitivity, f1, auc

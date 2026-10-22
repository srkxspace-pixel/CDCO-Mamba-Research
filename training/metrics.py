import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score
import torch

def calculate_metrics(pred, label):
    pred_scores = pred
    pred = np.argmax(pred, axis=-1)
    label = np.argmax(label, axis=-1)
    conf_matrix = confusion_matrix(label, pred, labels=list(range(np.shape(pred_scores)[-1])))  # rows true, cols predicted
    accuracy = accuracy_score(label, pred)
    # upstream swapped FP and FN (its SEN was precision) and averaged over both classes, so binary SPE == SEN
    tp = np.diag(conf_matrix)
    fn = conf_matrix.sum(1) - tp
    fp = conf_matrix.sum(0) - tp
    tn = conf_matrix.sum() - tp - fn - fp
    with np.errstate(invalid='ignore', divide='ignore'):
        if len(tp) == 2:  # binary: class 1 (patient, e.g. ASD) is the positive class
            avg_specificity, avg_sensitivity = tn[1] / (tn[1] + fp[1]), tp[1] / (tp[1] + fn[1])
        else:  # one-vs-rest, macro-averaged
            avg_specificity, avg_sensitivity = np.mean(tn / (tn + fp)), np.mean(tp / (tp + fn))
    f1 = f1_score(label, pred, average='macro')
    # AUC from predicted probabilities (upstream binarised the argmax, i.e. AUC of hard labels)
    probs = torch.softmax(torch.tensor(np.asarray(pred_scores, dtype=np.float64)), -1).numpy()
    if probs.shape[1] == 2:
        auc = roc_auc_score(label, probs[:, 1])
    else:
        auc = roc_auc_score(label, probs, average='macro', multi_class='ovr', labels=list(range(probs.shape[1])))
    return accuracy, avg_specificity, avg_sensitivity, f1, auc


if __name__ == '__main__':
    # 3 controls (class 0) + 3 patients (class 1): one control called patient, two patients called control
    label = np.eye(2)[[0, 0, 0, 1, 1, 1]]
    pred = np.eye(2)[[0, 0, 1, 0, 0, 1]] * 4.0
    acc, spe, sen, f1, auc = calculate_metrics(pred, label)
    assert np.isclose(acc, 0.5) and np.isclose(spe, 2 / 3) and np.isclose(sen, 1 / 3), (acc, spe, sen)
    print('metric checks passed: SPE', spe, 'SEN', sen)

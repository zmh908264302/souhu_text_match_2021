import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, classification_report
from sklearn.metrics import precision_recall_fscore_support


def evaluate(y_pred, y_true):
    # y_pred_softmax = F.log_softmax(y_pred, dim=1)
    _, y_pred = torch.max(y_pred.data, dim=1)
    # y_pred = F.log_softmax(y_pred)[:, 1]
    # y_pred = y_pred > threshold
    y_pred = y_pred.numpy()
    y_true = y_true.numpy()

    # f1 = f1_score(y_true, y_pred, average="macro")
    prf = precision_recall_fscore_support(y_true, y_pred, average="macro")
    correct = np.sum((y_true == y_pred).astype(int))
    acc = correct / y_pred.shape[0]
    return acc, prf


def evaluate_pos(start_poses, start_logits, end_poses, end_logits):
    # [batch_size, sequence_length]
    _, start_pred = torch.max(start_logits.data, dim=1)
    start_pred = start_pred.numpy()
    start_pos = start_poses.numpy()

    _, end_pred = torch.max(end_logits.data, dim=1)
    end_pred = end_pred.numpy()
    end_pos = end_poses.numpy()

    f1_start_pos = f1_score(start_pos, start_pred, average="micro")
    f1_end_pos = f1_score(end_pos, end_pred, average="micro")

    correct_start_pos = np.sum((start_pos == start_pred).astype(int))
    correct_end_pos = np.sum((end_pos == end_pred).astype(int))

    acc_start_pos = correct_start_pos / start_pred.shape[0]
    acc_end_pos = correct_end_pos / end_pred.shape[0]

    # 起或止位置预测准确率；起和止位置均预测准确率
    start_or_end_crt_acc, start_and_end_crt_acc = correct_start2end_pos(start_pos, start_pred, end_pos, end_pred)

    return acc_start_pos, f1_start_pos, acc_end_pos, f1_end_pos, start_or_end_crt_acc, start_and_end_crt_acc


def correct_start2end_pos(start_pos: np.ndarray, start_pred: np.ndarray, end_pos, end_pred):
    """起止位置准确率计算
    分两种情况：
    1. 起止任一位置正确
    2. 起止位置均准确
    """
    start_judge = start_pos == start_pred
    end_judge = end_pos == end_pred
    start_or_end_crt_cnt, start_and_end_crt_cnt = 0, 0
    for step, (i, j) in enumerate(zip(start_judge, end_judge)):
        if i or j:
            start_or_end_crt_cnt += 1
        if i and j:
            start_and_end_crt_cnt += 1
    num = len(start_pos)  # batch size
    return start_or_end_crt_cnt / num, start_and_end_crt_cnt / num


def get_best_threshold(y_pred, y_true):
    best_acc, best_f1 = 0, 0
    best_threshold = 0
    thresholds = [0.01 * i for i in range(100)]
    for threshold in thresholds:
        acc, f1 = evaluate(y_pred, y_true, threshold)
        if f1 >= best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_acc = acc
    return best_acc, best_f1, best_threshold


def class_report(y_pred, y_true, threshold):
    # y_pred = F.log_softmax(y_pred)[:, 1]
    # y_pred = y_pred > threshold
    y_pred_softmax = F.log_softmax(y_pred)
    _, y_pred = torch.max(y_pred_softmax.data, dim=1)
    y_pred = y_pred.numpy()
    y_true = y_true.numpy()

    classify_report = classification_report(y_true, y_pred)
    print('\n\nclassify_report:\n', classify_report)

# The original code only outputs the accuracy and the loss.
# Process the file model_output.tsv and calculate precision, recall, and F1 for each class
import argparse
import os
from typing import List
import numpy as np

def compute_recall(predictions: List[str], labels: List[str], target_class: str = "C") -> float:
    correct = 0
    total = 0
    if target_class == "C":
        negative_class = "N"
    else:
        assert target_class == "N", "Target class must be one of C or N"
        negative_class = "C"
    for pred, label in zip(predictions, labels):
        if pred == target_class and label == target_class: # True positives
            correct += 1
            total += 1
        if pred == negative_class and label == target_class: # False negatives
            total += 1
    return correct / total * 100

def compute_precision(predictions: List[str], labels: List[str], target_class: str = "C") -> float:
    correct = 0
    total = 0
    if target_class == "C":
        negative_class = "N"
    else:
        assert target_class == "N", "Target class must be one of C or N"
        negative_class = "C"
    for pred, label in zip(predictions, labels):
        if pred == target_class and label == target_class: # True positives
            correct += 1
            total += 1
        if pred == target_class and label == negative_class: # False positives
            total += 1
    return correct / total * 100


def compute_F_measure(recall: float, precision: float, beta: float = 1) -> float:
    return (beta ** 2 + 1) * precision * recall / (beta ** 2 * (precision + recall))

if __name__ == '__main__':
    pred_file = os.path.join(os.getcwd(), "experiments/base_model/model_output.tsv")
    with open(pred_file, 'r') as f:
        lines = f.readlines()
    lines = np.array([np.array(line.split()) for line in lines if len(line.split()) > 1])

    for target_class in ["C", "N"]:
        precision = compute_precision(lines.T[2], lines.T[1], target_class)
        print(f"Model has a precision of {round(precision, 2)}% for class {target_class}")
        recall = compute_recall(lines.T[2], lines.T[1], target_class)
        print(f"Model has a recall of {round(recall, 2)}% for class {target_class}")
        Fmeasure = compute_F_measure(recall, precision)
        print(f"Model has a F1 measure of {round(Fmeasure, 2)}% for class {target_class}")

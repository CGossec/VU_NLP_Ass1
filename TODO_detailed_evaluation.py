# The original code only outputs the accuracy and the loss.
# Process the file model_output.tsv and calculate precision, recall, and F1 for each class
import argparse
import os
from typing import List
import numpy as np

def compute_recall(predictions: List[str], labels: List[str]) -> float:
    correct = 0
    total = 0
    for pred, label in zip(predictions, labels):
        if pred == "C" and label == "C": # True positives
            correct += 1
            total += 1
        if pred == "N" and label == "C": # False negatives
            total += 1
    return correct / total * 100

def compute_precision(predictions: List[str], labels: List[str]) -> float:
    correct = 0
    total = 0
    for pred, label in zip(predictions, labels):
        if pred == "C" and label == "C": # True positives
            correct += 1
            total += 1
        if pred == "C" and label == "N": # False positives
            total += 1
    return correct / total * 100


def compute_F_measure(recall: float, precision: float, beta: float = 1) -> float:
    return (beta ** 2 + 1) * precision * recall / (beta ** 2 * (precision + recall))

if __name__ == '__main__':
    pred_file = os.path.join(os.getcwd(), "experiments/base_model/model_output.tsv")
    with open(pred_file, 'r') as f:
        lines = f.readlines()
    lines = np.array([np.array(line.split()) for line in lines if len(line.split()) > 1])
    precision = compute_precision(lines.T[2], lines.T[1])
    print(f"Model has a precision of {round(precision, 2)}%")
    recall = compute_recall(lines.T[2], lines.T[1])
    print(f"Model has a recall of {round(recall, 2)}%")
    Fmeasure = compute_F_measure(recall, precision)
    print(f"Model has a F1 measure of {round(Fmeasure, 2)}%")

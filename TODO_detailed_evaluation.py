# The original code only outputs the accuracy and the loss.
# Process the file model_output.tsv and calculate precision, recall, and F1 for each class
import os

def compute_recall() -> float:
    pass

def compute_precision() -> float:
    pass

def compute_F_measure(recall: float, precision: float, beta: float = 1) -> float:
    return (beta ** 2 + 1) * precision * recall / (beta ** 2 * (precision + recall))

if __name__ == '__main__':
    pred_file = os.path.join(os.getcwd(), "experiments/base_model/model_output.tsv")

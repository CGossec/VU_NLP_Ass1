# This file compares performance for different values of a single hyperparameter and plots the result
# Because a lot of stuff is done in train.py's main file, this file will not work without uncommenting L111-112
import argparse
import logging
import os

import numpy as np
import torch
import torch.optim as optim
from tqdm import trange

import utils
import model.net as net
from model.data_loader import DataLoader
from evaluate import evaluate, evaluate_and_output
from train import train_and_evaluate
from TODO_detailed_evaluation import compute_recall, compute_precision, compute_F_measure

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/preprocessed',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/pipeline',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")

def train_main(args, params):
    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # load data
    data_loader = DataLoader(args.data_dir, params)
    data = data_loader.load_data(['train', 'val'], args.data_dir)
    train_data = data['train']
    val_data = data['val']

    # specify the train and val dataset sizes
    params.train_size = train_data['size']
    params.val_size = val_data['size']

    logging.info("- done.")

    # Define the model and optimizer
    model = net.Net(params).cuda() if params.cuda else net.Net(params)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # fetch loss function and metrics
    loss_fn = net.loss_fn
    metrics = net.metrics

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_data, val_data, optimizer, loss_fn, metrics, params, args.model_dir,
                       None, data_loader)

def evaluate_main(args, params):
    """
        Evaluate the model on the test set.
    """
    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # load data
    data_loader = DataLoader(args.data_dir, params)
    data = data_loader.load_data(['test'], args.data_dir)
    test_data = data['test']

    # specify the test set size
    params.test_size = test_data['size']
    test_data_iterator = data_loader.data_iterator(test_data, params)

    logging.info("- done.")

    # Define the model
    model = net.Net(params).cuda() if params.cuda else net.Net(params)

    loss_fn = net.loss_fn
    metrics = net.metrics

    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    num_steps = (params.test_size + 1) // params.batch_size
    # MY ADJUSTMENTS
    # reverse the vocab and tag dictionary to be able to map back from ids to words and tags
    id2word = {v: k for k, v in data_loader.vocab.items()}
    tags = {v: k for k, v in data_loader.tag_map.items()}
    outfile = args.model_dir + "/model_output.tsv"
    test_metrics = evaluate_and_output(model, loss_fn, test_data_iterator, metrics, num_steps, id2word, tags, outfile)
    save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)


def weighted_F_measure(args):
    pred_file = os.path.join(os.getcwd(), f"{args.model_dir}/model_output.tsv")
    with open(pred_file, 'r') as f:
        lines = f.readlines()
    lines = np.array([np.array(line.split()) for line in lines if len(line.split()) > 1])

    Fmeasure = {}
    for target_class in ["C", "N"]:
        precision = compute_precision(lines.T[2], lines.T[1], target_class)
        recall = compute_recall(lines.T[2], lines.T[1], target_class)
        Fmeasure[target_class] = compute_F_measure(recall, precision)

    nb_class_C = 0
    nb_class_N = 0
    for label in lines.T[1]:
        if label == "C":
            nb_class_C += 1
        else:
            nb_class_N += 1

    weighted_F1 = (Fmeasure["C"] * nb_class_C + Fmeasure["N"] * nb_class_N) / (nb_class_C + nb_class_N)
    return weighted_F1


if __name__ == "__main__":
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()     # use GPU is available

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)

    values = [1e-5, 1e-4, 1e-3, 5e-3, 1e-2]
    values_to_plot = []
    for value in values:
        params.learning_rate = value
        train_main(args, params)
        evaluate_main(args, params)
        values_to_plot.append(weighted_F_measure(args))
        os.remove(os.path.join(os.getcwd(), args.model_dir, "best.pth.tar"))
        os.remove(os.path.join(os.getcwd(), args.model_dir, "last.pth.tar"))
    fig, ax = plt.subplots()
    ax.plot(np.array(values), values_to_plot)
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("F1 score")
    plt.show()
# Implement four baselines for the task.
# Majority baseline: always assigns the majority class of the training data
# Random baseline: randomly assigns one of the classes. Make sure to set a random seed and average the accuracy over 100 runs.
# Length baseline: determines the class based on a length threshold
# Frequency baseline: determines the class based on a frequency threshold

from model.data_loader import DataLoader
import random
random.seed(10)
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from typing import List, Tuple

# Each baseline returns predictions for the test data. The length and frequency baselines determine a threshold using the development data.
def split(word: str) -> List[str]:
    return [char for char in word]

def preprocessing(testinput: List[str], testlabels: List[str]) -> pd.DataFrame:
    'Preprocess and return a pandas df'
    label_arr = []
    token_arr = []
    for instance, labels in zip(testinput,testlabels):
        tokens = instance.split(" ")
        labels = split(labels.replace(' ', ''))
        for token, label in zip(tokens, labels):
            token_arr.append(token)
            label_arr.append(label)
        token_arr.append("--")
        label_arr.append("--")
    df = pd.DataFrame(list(zip(token_arr, label_arr)),
               columns =['Tokens', 'Labels'])
    return df


def majority_baseline(train_sentences: List[str], train_labels: List[str], testinput: List[str], testlabels: List[str]) -> Tuple[float, pd.DataFrame]:
    # TODO: determine the majority class based on the training data
    # ...

    df = preprocessing(testinput, testlabels)
    majority_class = df['Labels'].mode()[0]

    df["prediction"] = majority_class
    df["prediction"] = np.where(df["Labels"] == "--", "--", df["prediction"])

    # TODO: calculate accuracy for the test input
    # ...
    accuracy = accuracy_score(df["Labels"], df["prediction"])
    return accuracy, df


def random_baseline(train_sentences: List[str], train_labels: List[str], testinput: List[str], testlabels: List[str]) -> Tuple[float, pd.DataFrame]:
    # TODO: determine the majority class based on the training data
    # ...
    df = preprocessing(testinput, testlabels)

    rand_list = df["Labels"].unique()
    rand_list = rand_list.tolist()
    rand_list.remove('--')

    def random_class(row: pd.Series) -> str:
        rand_value = random.choice(rand_list)
        return rand_value

    df["prediction"] = df["Labels"].apply(random_class)
    df["prediction"] = np.where(df["Labels"]=="--", "--", df["prediction"])

    # TODO: calculate accuracy for the test input
    # ...
    accuracy = accuracy_score(df["Labels"], df["prediction"])
    return accuracy, df


def length_baseline(train_sentences: List[str], train_labels: List[str], testinput: List[str], testlabels: List[str]) -> Tuple[float, pd.DataFrame]:
    # TODO: determine the majority class based on the training data
    # ...
    length = 7
    df = preprocessing(testinput, testlabels)

    df["prediction"] = np.where(df["Tokens"].str.len() > length, "C", "N")
    df["prediction"] = np.where(df["Labels"] == "--", "--", df["prediction"])

    # TODO: calculate accuracy for the test input
    # ...
    accuracy = accuracy_score(df["Labels"], df["prediction"])
    return accuracy, df

def frequency_baseline(train_sentences: List[str], train_labels: List[str], testinput: List[str], testlabels: List[str])  -> Tuple[float, pd.DataFrame]:
    # TODO: determine the majority class based on the training data
    # ...
    freq = 2
    df = preprocessing(testinput,testlabels)
    df["freq"] = df['Tokens'].map(df['Tokens'].value_counts())
    df["prediction"] = np.where(df["freq"] >= freq, "N", "C")
    df["prediction"] = np.where(df["Labels"] == "--", "--", df["prediction"])
    accuracy = accuracy_score(df["Labels"], df["prediction"])
    # TODO: calculate accuracy for the test input
    # ...
    #Before returning df I am deleting freq column for aligning other baselines
    df.drop('freq', inplace=True, axis=1)
    return accuracy, df


if __name__ == '__main__':
    train_path = "data/preprocessed/train/"
    dev_path = "data/preprocessed/val/"
    test_path = "data/preprocessed/test/"

    # Note: this loads all instances into memory. If you work with bigger files in the future, use an iterator instead.
    with open(train_path + "sentences.txt", encoding='utf-8') as sent_file:
        train_sentences = sent_file.readlines()
    with open(train_path + "labels.txt", encoding='utf-8') as label_file:
        train_labels = label_file.readlines()

    with open(dev_path + "sentences.txt", encoding='utf-8') as dev_file:
        dev_sentences = dev_file.readlines()
    with open(train_path + "labels.txt", encoding='utf-8') as dev_label_file:
        dev_labels = dev_label_file.readlines()

    with open(test_path + "sentences.txt", encoding='utf-8') as testfile:
        testinput = testfile.readlines()
    with open(test_path + "labels.txt", encoding='utf-8') as test_label_file:
        testlabels = test_label_file.readlines()

    #Returning result dfs and acc for both test and dev
    majority_accuracy_test, majority_predictions_test_df = majority_baseline(train_sentences, train_labels, testinput, testlabels)
    majority_accuracy_dev, majority_predictions_dev_df = majority_baseline(train_sentences, train_labels, dev_sentences, dev_labels)

    random_accuracy_test, random_predictions_test_df = random_baseline(train_sentences, train_labels, testinput, testlabels)
    random_accuracy_dev, random_predictions_dev_df = random_baseline(train_sentences, train_labels, dev_sentences, dev_labels)

    length_accuracy_test, length_predictions_test_df = length_baseline(train_sentences, train_labels, testinput, testlabels)
    length_accuracy_dev, length_predictions_dev_df = length_baseline(train_sentences, train_labels, dev_sentences, dev_labels)

    freq_accuracy_test, freq_predictions_test_df = frequency_baseline(train_sentences, train_labels, testinput, testlabels)
    freq_accuracy_dev, freq_predictions_dev_df = frequency_baseline(train_sentences, train_labels, dev_sentences, dev_labels)

    # TODO: output the predictions in a suitable way so that you can evaluate them
    print("majority_accuracy_test", majority_accuracy_test)
    print("majority_accuracy_dev", majority_accuracy_dev)
    print("random_accuracy_test", random_accuracy_test)
    print("random_accuracy_dev", random_accuracy_dev)
    print("length_accuracy_test", length_accuracy_test)
    print("length_accuracy_dev", length_accuracy_dev)
    print("freq_accuracy_test", freq_accuracy_test)
    print("freq_accuracy_dev", freq_accuracy_dev)

import spacy
import os
import pandas as pd

df_old = pd.read_csv('data/original/english/WikiNews_Train.tsv', sep='\t',header=None)
nlp = spacy.load("en_core_web_sm")


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ extract basic statistics (7) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Extracting the columns Target word (5), binary label(10) and probabilistic label(11)
df = df_old[[4,9,10]]
df.columns = ["Target word", "Binary", "Probabilistic"]

print(f"Total number of instances labeled 0 and 1 in doc are: {df['Binary'].value_counts()}")
print(f"Min, max, median, mean, and stdev of the probabilistic label in doc are: {df['Probabilistic'].min()}, "
      f"{df['Probabilistic'].max()}, {df['Probabilistic'].median()}, {df['Probabilistic'].mean()} "
      f"and {df['Probabilistic'].std()}.")

counter = 0
max = 0
for row in range(len(df)):
    doc = nlp(df.loc[row, "Target word"])
    counter_row = 0
    for token in doc:
        counter_row = counter_row + 1
        if counter_row > max:
            max = counter_row
    if counter_row > 1:
        counter = counter + 1

print(f"Number of instances consisting of more than one token are: {counter}")
print(f"Maximum number of tokens for an instance is: {max}")







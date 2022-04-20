import spacy
import os
import pandas as pd

df_old = pd.read_csv('data/original/english/WikiNews_Train.tsv', sep='\t',header=None)
nlp = spacy.load("en_core_web_sm")


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ extract basic statistics (7) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Extracting the columns Target word (5), binary label(10) and probabilistic label(11)
df = df_old[[4,9,10]]
print(df)


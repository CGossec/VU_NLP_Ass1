import spacy
import pandas as pd
from wordfreq import word_frequency
import matplotlib.pyplot as plt

df_old = pd.read_csv('data/original/english/WikiNews_Train.tsv', sep='\t',header=None)
nlp = spacy.load("en_core_web_sm")


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ extract basic statistics (7) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Extracting the columns Target word (5), binary label(10) and probabilistic label(11)
df = df_old[[4,9,10]]
df.columns = ["Target word", "Binary", "Probabilistic"]

print(f"Total number of instances labeled 0 and 1 in doc are: {df['Binary'].value_counts()}")
print(f"Min, max, median, mean, and stdev of the probabilistic label in doc are: {df['Probabilistic'].min()}, "
      f"{df['Probabilistic'].max()}, {df['Probabilistic'].median()}, {df['Probabilistic'].mean()} "
      f"and {df['Probabilistic'].std()}.")

counter = 0
max = 0

data=[] # list for the instances of single tokens for the Target word
# for row in range(len(df)):
#     doc = nlp(df.loc[row, "Target word"])
#     counter_row = 0
#     for token in doc:
#         counter_row = counter_row + 1
#         if counter_row > max:
#             max = counter_row
#     if counter_row > 1:
#         counter = counter + 1
#     else:
#         data.append(df.iloc[row])

print(f"Number of instances consisting of more than one token are: {counter}")
print(f"Maximum number of tokens for an instance is: {max}")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Explore linguistic characteristics (8) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# we will focus on only the instances with one token, so here we create a different dataset based on the 'data' list

# df_single = pd.DataFrame(data)
# df_single.to_csv('data/original/english/WikiNews_Train_single.tsv', sep='\t',header=None, index=False)

df_single = pd.read_csv('data/original/english/WikiNews_Train_single.tsv', sep='\t')
# df_single.columns = ["Target word", "Binary", "Probabilistic"]
# length = []
# frequency = []
# POS_tag = []
# for row in range(len(df_single)):
#     word = nlp(df_single.loc[row, "Target word"])
#     length.append(len(word.text))
#     frequency.append(word_frequency(word.text, 'en'))
#     for token in word:  # there is only 1 word, but you have to call each model for .tag_
#         POS_tag.append(token.tag_)
#
# df_single['new_col0'] = length
# df_single['new_col1'] = frequency
# df_single['new_col2'] = POS_tag
# df_single.to_csv('data/original/english/WikiNews_Train_single.tsv', sep='\t',header=None, index=False)

df_single.columns = ["Target word", "Binary", "Probabilistic","Length", "Frequency", "POS-tag"]


print(f"The Pearson correlation of length and complexity is: "
      f"{df_single['Length'].corr(df_single['Probabilistic'])}")
print(f"The Pearson correlation of frequency and complexity is: "
      f"{df_single['Frequency'].corr(df_single['Probabilistic'])}")
print(f"scatterplot complexity and length")
plt.scatter(data=df_single, x='Length', y='Probabilistic')
plt.xlabel("Length")
plt.ylabel("Probability of complexity")
plt.show()

print(f"scatterplot complexity and frequency")
plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})
plt.scatter(data=df_single, x='Frequency', y='Probabilistic')
plt.xlabel("Frequency")
plt.ylabel("Probability of complexity")
plt.show()

print(f"scatterplot complexity and POS-tag")
plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})
plt.scatter(data=df_single, x='POS-tag', y='Probabilistic')
plt.xlabel("POS-tag (finegrained)")
plt.ylabel("Probability of complexity")
plt.show()




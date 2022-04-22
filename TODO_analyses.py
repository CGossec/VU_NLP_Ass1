# Implement linguistic analyses using spacy
# Run them on data/preprocessed/train/sentences.txt
import spacy
import os
import pandas as pd
from wordfreq import word_frequency
import matplotlib.pyplot as plt

nlp = spacy.load("en_core_web_sm")
doc = nlp(open(os.path.join(os.getcwd(), "data/preprocessed/train/sentences.txt"),encoding="utf8").read())

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Basic tokenization ~~~~~~~~~~~~~~~~~~~~~~

nb_tokens = 0
nb_words = 0
nb_chars = 0
pos_frequencies = {}
for token in doc:
    nb_tokens += 1
    if not token.is_punct:
        nb_words += 1
        nb_chars += len(token.text)
    if token.pos_ not in pos_frequencies:
        pos_frequencies[token.pos_] = 0
    pos_frequencies[token.pos_] += 1

print(f"Total number of tokens in doc is: {nb_tokens}")
print(f"Total number of types in doc is: {len(nlp.vocab)}")
print(f"Total number of words in doc is: {nb_words}")
print(f"There are {len([x for x in doc.sents])} sentences in the document, so the average "
    f"number of words per sentence is: {nb_words} / {len([x for x in doc.sents])} = {round(nb_words / len([x for x in doc.sents]),2)}")
print(f"There are {nb_words} words and a total of {nb_chars} letters in those words; so the average "
    f"number of letters per word is: {nb_chars} / {nb_words} = {round(nb_chars / nb_words,2)}\n")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Word classes ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pos_frequencies_top_10 = {pos_tag: freq for pos_tag, freq in sorted(pos_frequencies.items(), key=lambda item:item[1], reverse=True)[:10]}
print(pos_frequencies_top_10)

word_frequencies = {x: {} for x in pos_frequencies_top_10.keys()}
for token in doc:
    if token.pos_ in word_frequencies.keys():
        if token.text not in word_frequencies[token.pos_]:
            word_frequencies[token.pos_][token.text] = 0
        word_frequencies[token.pos_][token.text] += 1

for key in word_frequencies.keys():
    print(f"Most occurences of part-of-speech {key}:")
    print({word: freq for word, freq in sorted(word_frequencies[key].items(), key=lambda item:item[1], reverse=True)[:3]})
    print(f"Least frequent word of part-of-speech {key}:")
    print({word: freq for word, freq in sorted(word_frequencies[key].items(), key=lambda item:item[1])[:1]})

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ N-grams ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

bigrams = {}
trigrams = {}
prev_token = None
prev_prev_token = None
for token in doc:
    bigram = [prev_token, token]
    if str(bigram) not in bigrams:
        bigrams[str(bigram)] = 0
    bigrams[str(bigram)] += 1

    trigram = [prev_prev_token, prev_token, token]
    if str(trigram) not in trigrams:
        trigrams[str(trigram)] = 0
    trigrams[str(trigram)] += 1

    prev_prev_token = prev_token
    prev_token = token

bigrams_top_3 = {ngram: freq for ngram, freq in sorted(bigrams.items(), key=lambda item:item[1], reverse=True)[:3]}
print(f"Top 3 bigrams are: {bigrams_top_3}")
trigrams_top_3 = {ngram: freq for ngram, freq in sorted(trigrams.items(), key=lambda item:item[1], reverse=True)[:3]}
sentences=[x for x in doc.sents]
print(f"Top 3 trigrams are: {trigrams_top_3}")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ POS N-grams ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

bigrams = {}
trigrams = {}
prev_token = None
prev_prev_token = None
for token in doc:
    bigram = [prev_token, token.pos_]
    if str(bigram) not in bigrams:
        bigrams[str(bigram)] = 0
    bigrams[str(bigram)] += 1

    trigram = [prev_prev_token, prev_token, token.pos_]
    if str(trigram) not in trigrams:
        trigrams[str(trigram)] = 0
    trigrams[str(trigram)] += 1

    prev_prev_token = prev_token
    prev_token = token.pos_

bigrams_top_3 = {ngram: freq for ngram, freq in sorted(bigrams.items(), key=lambda item:item[1], reverse=True)[:3]}
print(f"Top 3 POS-bigrams are: {bigrams_top_3}")
trigrams_top_3 = {ngram: freq for ngram, freq in sorted(trigrams.items(), key=lambda item:item[1], reverse=True)[:3]}
print(f"Top 3 POS-trigrams are: {trigrams_top_3}")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Lemmatization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

lemmas_occ = {}
for token in doc:
    if token.lemma_ not in lemmas_occ:
        lemmas_occ[token.lemma_] = set([token.text])
    else:
        lemmas_occ[token.lemma_].add(token.text)
        if len(lemmas_occ[token.lemma_]) > 5:
            print(f"Lemma '{token.lemma_}' appeared as {lemmas_occ[token.lemma_]}")
            break

inflections = {inf: False for inf in lemmas_occ[token.lemma_]}
for sent in doc.sents:
    for inflection in inflections.keys():
        if not inflections[inflection]:
            if inflection in sent.text:
                print(f"A sentence containing the inflection '{inflection}' was '{sent.text}'")
                inflections[inflection] = True

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Named entity recognition ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

nb_named_entities = len(doc.ents)
diff_entities = set()
for ent in doc.ents:
    if ent not in diff_entities:
        diff_entities.add(ent.label_)
print(f"Number of named entities: {nb_named_entities}")
print(f"Number of different entity labels: {len(diff_entities)}")


first_five_sent = "".join([sent.text for sent in doc.sents][:5])
print(first_five_sent)

count = 0
for entity in doc.ents:
    if count >= 7: # Manually counted the number of entities detected by our model on the first five sentences
        break
    count += 1
    print(f"{entity.text} ({entity.label_})")
# Entities do not look like they have been properly recognized.
# Ages of the children are recognized as dates, when they should
# probably be recognized as cardinal numbers. ROS, which seemingly is the component of a
# microscopic study involving fungi, should also not be recognized as a GeoPolitical Entity and
# simply be ignored. Finally, a backslash is also erroneously recognized as ORG, when it is only
# an escape character and should be ignored. The word "Police" is also not detected at all, but
# should be classified as GPE


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

data=[] # list for the instances of single tokens for the Target word, only used for creating dataset
# Counting number of tokens and determining the max token length
for row in range(len(df)):
    doc = nlp(df.loc[row, "Target word"])
    counter_row = 0
    for token in doc:
        counter_row = counter_row + 1
        if counter_row > max:
            max = counter_row
    if counter_row > 1:
        counter = counter + 1
#     else:
#         data.append(df.iloc[row])

print(f"Number of instances consisting of more than one token are: {counter}")
print(f"Maximum number of tokens for an instance is: {max}")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Explore linguistic characteristics (8) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# we will focus on only the instances with one token, so here we create a different dataset based on the 'data' list

# df_single = pd.DataFrame(data)
# df_single.to_csv('data/original/english/WikiNews_Train_single.tsv', sep='\t',header=None, index=False)

df_single = pd.read_csv('data/original/english/WikiNews_Train_single.tsv', sep='\t')

# Creating datafile with the columns: length, frequency and POS-tag. It is commented out as the file is already created

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

# Calculating the Pearson correlation

print(f"The Pearson correlation of length and complexity is: "
      f"{df_single['Length'].corr(df_single['Probabilistic'])}")
print(f"The Pearson correlation of frequency and complexity is: "
      f"{df_single['Frequency'].corr(df_single['Probabilistic'])}")

# Forming the scatterplots
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




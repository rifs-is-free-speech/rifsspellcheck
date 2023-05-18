import os
import re
import nltk
import random
import pandas as pd

from glob import glob
from num2words import num2words
from spello.model import SpellCorrectionModel  
from halo import Halo

spinner = Halo(text='Loading files', spinner='bouncingBall')

path = input('Provide path to corpus')

def bigcorpus():
    """Generator to read the corpus file"""
    filenames = glob(path + "/**", recursive=True)
    random.shuffle(filenames)
    for filename in filenames:
        if os.path.basename(filename) in ["README.txt", "LICENSE"]: continue
        if os.path.isdir(filename): continue
        if os.path.splitext(filename)[1] == ".jsonl": continue
        with open(filename, "r") as f:
            text = f.read()  
        sent_text = nltk.sent_tokenize(text) 
        for sent in sent_text:
            try:
                sent = re.sub(r'\d+', lambda x: num2words(int(x.group(0)),lang="dk").lower(), sent)
            except AttributeError:
                continue

            yield re.sub('\W+',' ', sent)

print("Loading words")

correct_words = []

words = pd.read_csv("da_words.csv", sep="\t", low_memory=False)

with open("20200409-valid-Danish-first-names.txt", "r") as f:
    for line in f:
        correct_words.append(line.strip().lower())

with open("20200419-Danish-words.txt", "r") as f:
    for line in f:
        correct_words.append(line.strip().lower())

for i, row in words.iterrows():
    if row["word"]:
        if isinstance(row["word"], str):
            correct_words.append(row["word"].lower())
    if row["alternatives"]:
        if isinstance(row["alternatives"], str):
            alt = row["alternatives"].split(",")
            for word in alt:
                correct_words.append(word.lower())
    if row["strong_declension"]:
        if isinstance(row["strong_declension"], str):
            strong = row["strong_declension"].split(",")
            for word in strong:
                correct_words.append(word.lower())

correct_words = set(correct_words)

sp = SpellCorrectionModel(language='en')

generator = bigcorpus()

spinner.start()

sample = list(correct_words)
for i in range(2500000):
    sample.append(next(generator))
#sample = [sent for sent in generator]
spinner.stop()

print("Done!")

sp.train(sample)
sp.save(model_save_dir=".")

while True:
    print(sp.spell_correct(input("Enter a sentence: ")))

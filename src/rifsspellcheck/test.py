import re
import spacy
import contextualSpellCheck

import pandas as pd
from spello.model import SpellCorrectionModel

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


print("Loading BERT model")

nlp = spacy.load("da_core_news_sm")

nlp.add_pipe("contextual spellchecker", 
    config={
        "max_edit_dist": 500,
        "model_name": "Maltehb/danish-bert-botxo",
    },
)

print("Loading Spello model")

sp = SpellCorrectionModel(language='en')
sp.load("model.pkl")
sp.config.min_length_for_spellcorrection = 3
sp.config.max_length_for_spellcorrection = 30

df = pd.read_csv(input("Provide path to 'segments.csv' file"))

for i, row in df.iterrows():
    text = row["model_output"]
    print(f"Example: {i}")
    print(f"Raw:    {text}")
    print(f"Bert:   {nlp(text)._.outcome_spellCheck}")
    print(f"Spello: {sp.spell_correct(text)['spell_corrected_text']}")
    print(f"both:   {nlp(sp.spell_correct(text)['spell_corrected_text'])._.outcome_spellCheck}")
    new_text = []
    for word in text.split(" "):
        word = re.sub('\W+', '*', word)
        if word not in correct_words:
            new_text.append("[MASK]")
        else:
            new_text.append(word)
    new_text = " ".join(new_text)
    print(f"Masked:  {new_text}")
    print(f"After:   {nlp(new_text)._.outcome_spellCheck}")
    print("")

#df.to_csv("segments_spellchecked.csv", index=False)


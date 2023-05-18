import os
import re
import soundfile as sf
import librosa
import pandas as pd

from spello.model import SpellCorrectionModel
from shutil import copy2

class Predictor:
    def __init__(self, model):

        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        import torch

        self.model = Wav2Vec2ForCTC.from_pretrained(
            model
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.processor = Wav2Vec2Processor.from_pretrained(
            model
        )
        self.model.eval()


    def predict(self, record):
        import torch
        import re

        data = torch.tensor(record)
        input_dict = self.processor(
            data,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            logits = self.model(
                input_dict["input_values"]
                .squeeze(1)
                .to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            ).logits
        pred_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(pred_ids)[0]

        regex = r"\[UNK\]unk\[UNK\]|\[UNK]"
        transcription = re.sub(regex, "", transcription)
        return transcription


def load_spellchecker():
    spellfolder = os.path.dirname(os.path.realpath(__file__))
    
    correct_words = []
    words = pd.read_csv(f"{spellfolder}/da_words.csv", sep="\t", low_memory=False)
    with open(f"{spellfolder}/20200409-valid-Danish-first-names.txt", "r") as f:
        for line in f:
            correct_words.append(line.strip().lower())

    with open(f"{spellfolder}/20200419-Danish-words.txt", "r") as f:
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
    sp.load(f"{spellfolder}/../../models/model.pkl")
    sp.config.min_length_for_spellcorrection = 3
    sp.config.max_length_for_spellcorrection = 30
    return sp

def align_rifs(left, right) -> None:
    import pandas as pd
    from kaldialign import align

    EPS = "*"

    alignment = align(left, right, EPS)
    left = "".join([x[0] if x[1] != " " else " " for x in alignment])
    right = "".join([x[1] if x[0] != " " else " " for x in alignment])

    return (left, right)

def label_dataset(source, target, alignment_folder = "alignments", filename = "segments.csv", model= "Alvenir/wav2vec2-base-da-ft-nst"):
    print("Creating dataset")
    os.makedirs(target, exist_ok = True)
    os.makedirs(os.path.join(target, alignment_folder), exist_ok = True)

    print("Loading model")
    model = Predictor(model=model)
    print("Loading spellchecker")
    sp = load_spellchecker()

    mydir = os.path.join(source, alignment_folder)

    max_depth = 0
    bottom_most_dirs = []
    for dirpath, dirnames, filenames in os.walk(mydir):
        depth = len(dirpath.split(os.sep))
        if max_depth < depth:
            max_depth = depth
            bottom_most_dirs = [dirpath]
        elif max_depth == depth:
            bottom_most_dirs.append(dirpath)
    for folder in bottom_most_dirs:
        new_target = os.path.join(target, os.path.relpath(folder, source))
        os.makedirs(new_target, exist_ok = True)
        
        print(os.path.basename(folder))
        df = pd.read_csv(os.path.join(folder, filename))
        new_df_rows = []

        i, x = 0, 0
        for row in df.to_dict(orient="records"):
            i += 1
            wav = folder+"/"+row["file"]
            audio, sr = librosa.load(wav, sr=16_000, mono=True)
            prediction = model.predict(audio)
            text = row["model_output"]
            
            left, right = align_rifs(text, prediction)
            left_eps_ratio = left.count("*") / len(left)
            right_eps_ratio = right.count("*") / len(right)
            try:
                if left_eps_ratio < 0.03 and right_eps_ratio < 0.10:
                    if sp.spell_correct(text)["correction_dict"]:
                        continue
                    #print("left:   " + left)
                    #print("right:  " + right)
                    #print()
                    x += 1
                    new_file = os.path.join(new_target, row["file"])
                    copy2(wav, new_file)
                    new_df_rows.append(row)
            except KeyError:
                pass
        print(f"Only {x} out of {i} files were kept")
        pd.DataFrame(new_df_rows).to_csv(os.path.join(new_target, "segments.csv"), index=False)

if __name__ == "__main__":
    model = "/home/andst/hpc_city_gold"
    target = "/home/andst/Documents/rifs-is-free-speech/datasets/test/Den2Radio2/"
    source = "/home/andst/Documents/rifs-is-free-speech/datasets/Den2Radio/"
    alignments = "alignments"
    filename = "segments.csv"
    label_dataset(source, target, alignments, filename, model)


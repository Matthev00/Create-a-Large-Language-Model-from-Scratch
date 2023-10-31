import os
import lzma
from tqdm import tqdm
from pathlib import Path
import json
from typing import List, Tuple
import re


def xz_files_in_dir(directory):
    files = []
    for filename in os.listdir(directory):
        if filename.endswith(".xz") and os.path.isfile(
            os.path.join(directory, filename)
        ):
            files.append(filename)
    return files


def split_files(files, train_percent: float):
    """Calculate the split indices"""
    total_files = len(files)
    split_index = int(total_files * train_percent)  # 90% for training
    files_train = files[:split_index]
    files_val = files[split_index:]

    return files_train, files_val


def process_files(
    files_train,
    files_val,
    input_data="data/openwebtext",
    output_data_dir="data/preprocessed",
):
    # Process the files for training and validation separately
    output_file_train = os.path.join(output_data_dir, "train_split.txt")
    output_file_val = os.path.join(output_data_dir, "val_split.txt")
    vocab_file = os.path.join(output_data_dir, "vocab.txt")
    vocab = set()

    # Process the training files
    with open(output_file_train, "w", encoding="utf-8") as outfile:
        for filename in tqdm(files_train, total=len(files_train)):
            file_path = os.path.join(input_data, filename)
            with lzma.open(file_path, "rt", encoding="utf-8") as infile:
                text = infile.read()
                outfile.write(text)
                characters = set(text)
                vocab.update(characters)

    # Process the validation files
    with open(output_file_val, "w", encoding="utf-8") as outfile:
        for filename in tqdm(files_val, total=len(files_val)):
            file_path = os.path.join(input_data, filename)
            with lzma.open(file_path, "rt", encoding="utf-8") as infile:
                text = infile.read()
                outfile.write(text)
                characters = set(text)
                vocab.update(characters)

    # Write the vocabulary to vocab.txt
    with open(vocab_file, "w", encoding="utf-8") as vfile:
        for char in vocab:
            vfile.write(char + "\n")


def extract_qa_from_json(file_path: Path) -> str:
    with open(file_path, "r") as f:
        data = json.load(f)
    text = ""

    for sample in data:
        q = sample["question"]
        a = sample["answer"]
        text += q.capitalize() + " " + a.capitalize() + ". "
    return text


def get_paths(dir_path: Path) -> List[Path]:
    return [plik for plik in dir_path.iterdir() if plik.is_file()]


def split_text(text: str, train_split: float = 0.7) -> Tuple[str, str]:
    split_point = int(len(text) * train_split)
    return text[:split_point], text[split_point:]


def create_train_val_medical_data(dir_path: Path):
    input_dir = dir_path / "raw"
    output_dir = dir_path / "preprocessed"
    files = get_paths(dir_path=input_dir)

    text = ""
    for file_path in tqdm(files):
        text += extract_qa_from_json(file_path=file_path)

    train_text, val_text = split_text(text=text, train_split=0.8)
    # save train data
    train_file_path = output_dir / "train.txt"
    with open(train_file_path, "w", encoding="utf-8") as f:
        f.write(train_text)

    # save val data
    val_file_path = output_dir / "val.txt"
    with open(val_file_path, "w", encoding="utf-8") as f:
        f.write(val_text)


def main():
    dir_path = Path(
        "E:/projekty python/Create-a-Large-Language-Model-from-Scratch/data/finetuning_med"
    )
    create_train_val_medical_data(dir_path=dir_path)


if __name__ == "__main__":
    main()

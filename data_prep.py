import os
import lzma
from tqdm import tqdm


def xz_files_in_dir(directory):
    files = []
    for filename in os.listdir(directory):
        if filename.endswith(".xz") and os.path.isfile(os.path.join(directory, filename)): # noqa 5501
            files.append(filename)
    return files


def split_files(files,
                train_percent: float):
    """Calculate the split indices"""
    total_files = len(files)
    split_index = int(total_files * train_percent)  # 90% for training
    files_train = files[:split_index]
    files_val = files[split_index:]

    return files_train, files_val


def process_files(files_train,
                  files_val,
                  input_data="data/openwebtext",
                  output_data_dir="data/preprocessed"):
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
            vfile.write(char + '\n')


def main():
    folder_path = "E:/projekty python/Create-a-Large-Language-Model-from-Scratch/data/openwebtext" # noqa 5501
    files = xz_files_in_dir(folder_path)

    files_train, files_val = split_files(files=files,
                                         train_percent=0.9)

    process_files(files_train=files_train,
                  files_val=files_val,
                  input_data=folder_path)


if __name__ == "__main__":
    main()
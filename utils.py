import mmap
import random
import torch
from pathlib import Path
import argparse
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
from matplotlib import pyplot as plt
from typing import List


def prepare_vocab(file_path="data/preprocessed/vocab.txt"):
    """Returns Vocab size and decode/encode funcs"""
    with open("data/preprocessed/vocab.txt", "r", encoding="utf-8") as f:
        text = f.read()
        chars = sorted(list(set(text)))

    vocab_size = len(chars)

    str_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_ch = {i: ch for i, ch in enumerate(chars)}

    encode = lambda s: [str_to_int[c] for c in s]
    decode = lambda t: ''.join([int_to_ch[n] for n in t])

    return vocab_size, encode, decode


def get_finetunning_data(split,
                         block_size,
                         batch_size,
                         encode):
    filename = "data/TruthfulQA/finetune_info.jsonl" if split == 'train' else "data/TruthfulQA/finetune_info.jsonl" # noqa 5501
    text = ""
    start_index = len("{'prompt': 'Q: ")
    openings = []
    with open(filename, "r") as f:
        lines = f.readlines()

    for line in lines:
        if line.find('"completion": " yes"'):
            end_index = line.find("Helpful:")
            line = line[start_index:end_index]
            text += line.replace("A: ", "")
            openings.append(len(text))

    random_start = random.randint(0, len(text) - block_size*batch_size)
    start_pos = find_first_lower(arr=openings,
                                 index=random_start)
    end_pos = start_pos + block_size*batch_size
    sample = text[start_pos:end_pos]

    data = torch.tensor(encode(sample), dtype=torch.long)
    return data


def find_first_lower(arr: List,
                     index: int):
    arr = reversed(arr)
    for element in arr:
        if element < index:
            return element


def get_random_chunk(split,
                     block_size,
                     batch_size,
                     encode):
    filename = "data/preprocessed/train_split.txt" if split == 'train' else "data/preprocessed/val_split.txt" # noqa 5501
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # Determine the file size and a random position to start reading
            file_size = len(mm)
            start_pos = random.randint(0, (file_size) - block_size*batch_size)

            # Seek to the random position and read the block of text
            mm.seek(start_pos)
            block = mm.read(block_size*batch_size-1)

            # Decode the block to a string, ignoring any invalid byte sequences
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '') # noqa 5501

            # Train and test splits
            data = torch.tensor(encode(decoded_block), dtype=torch.long)

    return data


def get_batch(split,
              block_size,
              batch_size,
              encode_fn,
              device: torch.device = 'cuda:0',
              finetuning=False):
    if finetuning:
        data = get_finetunning_data(split=split,
                                    block_size=block_size,
                                    batch_size=batch_size,
                                    encode=encode_fn)
    else:
        data = get_random_chunk(split=split,
                                block_size=block_size,
                                batch_size=batch_size,
                                encode=encode_fn)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


def save_model(model,
               model_name):
    target_dir_path = Path("models")
    target_dir_path.mkdir(parents=True, exist_ok=True)

    model_save_path = target_dir_path / model_name

    torch.save(obj=model.state_dict(),
               f=model_save_path)


def parse_arguments():
    """
    Parse arguments:
    - bach_size
    - max_iters
    - lr (learning rate)
    """
    parser = argparse.ArgumentParser(
        description="Script trains LLM model(GPT architecture)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Size of Batch"
    )
    parser.add_argument(
        "--max_iters", type=int, default=1000, help="Range of training iterations" # noqa 5501
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4, help="Learning rate"
    )

    args = parser.parse_args()
    return args


def create_writer(experiment_name: str,
                  model_name: str,
                  extra: str = None) -> SummaryWriter:
    """
    Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir. # noqa 5501
    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra. # noqa 5501
    Where timestamp is the current date in YYYY-MM-DD format.

    Args:
        experiment_name (str): Nome of experiment.
        model_name (str): Name of the model.
        extra (str, optional): Anything extra to add int dir . Defaults to None.

    Returns:
        SummaryWriter: instance of a writer
    """
    timestamp = datetime.now().strftime("%Y-%m-%d")
    if extra:
        # Create log directory path
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra) # noqa 5501
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)

    return SummaryWriter(log_dir=log_dir)


def plot_loss_curves(results):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "test_loss": [...]
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()


def main():
    sample = get_finetunning_data(
        split="train",
        block_size=128,
        batch_size=32,
        encode="a")
    print(sample)
    print(len(sample))


if __name__ == "__main__":
    main()

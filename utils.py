import mmap
import random
import torch
from pathlib import Path
import argparse
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime


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
              device: torch.device = 'cuda:0'):
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
    - eval_steps
    """
    parser = argparse.ArgumentParser(
        description="Script trains LLM model(GPT architecture)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Size of Batch"
    )
    parser.add_argument(
        "--max_iters", type=int, default=1000, help="Range of training iterations"
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4, help="Learning rate"
    )
    parser.add_argument(
        "--eval_steps", type=int, default=200, help="Evaluation steps"
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

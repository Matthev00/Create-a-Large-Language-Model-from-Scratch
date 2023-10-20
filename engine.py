import torch
from torch.utils.tensorboard import SummaryWriter

from tqdm.auto import tqdm
from utils import get_batch


def train(model: torch.nn.Module,
          optimizer: torch.optim,
          writer: SummaryWriter,
          epochs: int,
          encode,
          device: torch.device = 'cuda:0',
          block_size: int = 128,
          batch_size: int = 32):
    """
    Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Stores evaluation metrics throughout.

    Args:
        model: A PyTorch model to be trained and tested.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        epochs: An integer indicating how many epochs to train for.
        device: A target device to compute on (e.g. "cuda" or "cpu").
        writer: SummaryWriter from TensorBoard, used to track experiments

    Returns:
        A dictionary of training and testing loss. Each metric has a value in a list for
        each epoch.
    """

    # Create empty results dict
    results = {"train_loss": [],
               "test_loss": []}

    for epoch in tqdm(range(epochs)):
        train_loss = train_step(model=model,
                                optimizer=optimizer,
                                encode=encode,
                                device=device,
                                block_size=block_size,
                                batch_size=batch_size)
        test_loss = test_step(model=model,
                              encode=encode,
                              device=device,
                              block_size=block_size,
                              batch_size=batch_size)

        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)

        writer.add_scalars(main_tag='Loss',
                           tag_scalar_dict={"train_loss": train_loss,
                                            "test_loss": test_loss},
                           global_step=epoch)

    writer.close()
    return results


def train_step(model: torch.nn.Module,
               optimizer: torch.optim,
               encode,
               device: torch.device = 'cuda:0',
               block_size: int = 128,
               batch_size: int = 32):
    """
    Trains a PyTorch model for single epoch

    Args:
        model (torch.nn.Module): model to be trained
        optimizer (torch.optim): optimizer
        device (torch.device, optional): target device, defaults to "cuda".
    """
    model.train()

    X, y = get_batch(split="train",
                     block_size=block_size,
                     batch_size=batch_size,
                     encode_fn=encode,
                     device=device)

    logits, loss = model(X, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return loss.item()


def test_step(model: torch.nn.Module,
              encode,
              device: torch.device = 'cuda:0',
              block_size: int = 128,
              batch_size: int = 32):

    model.eval()
    with torch.inference_mode():
        X, y = get_batch(split="test",
                         block_size=block_size,
                         batch_size=batch_size,
                         encode_fn=encode,
                         device=device)
        logits, loss = model(X, y)

    return loss.item()

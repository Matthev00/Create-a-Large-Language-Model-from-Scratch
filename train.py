import torch
from tqdm import tqdm
from utils import get_batch, prepare_vocab, save_model, parse_arguments
from model_builder import create_GPT_model


def estimate_loss(model,
                  eval_iters,
                  block_size,
                  batch_size,
                  encode_fn,
                  device):
    out = {}
    model.eval()
    with torch.inference_mode():
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in tqdm(range(eval_iters)):
                X, Y = get_batch(split=split,
                                 block_size=block_size,
                                 batch_size=batch_size,
                                 encode_fn=encode_fn,
                                 device=device)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
    model.train()
    return out


def main():
    args = parse_arguments()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = args.batch_size
    block_size = 128
    max_iters = args.max_iters
    learning_rate = args.lr
    eval_steps = args.eval_steps
    model_name = "GPT_Model_trained_5000_epochs.pth"

    vocab_size, encode, decode = prepare_vocab()

    model = create_GPT_model(vocab_size=vocab_size,
                             device=device)

    model.load_state_dict(torch.load(
        f="models/GPT_Model_trained_5000_epochs.pth",
        map_location=torch.device(device)))

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for i in tqdm(range(max_iters)):
        # test
        if i % eval_steps == 0:
            step_results = estimate_loss(model=model,
                                         eval_iters=eval_steps,
                                         block_size=block_size,
                                         batch_size=batch_size,
                                         encode_fn=encode,
                                         device=device)
            print(f"step: {i}, train loss: {step_results['train']:.3f}, val loss: {step_results['val']:.3f}") # noqa 5501

        # Train

        # get samle batch of data
        X, y = get_batch(split="train",
                         block_size=block_size,
                         batch_size=batch_size,
                         encode_fn=encode,
                         device=device)

        logits, loss = model(X, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(loss.item())

    save_model(model=model,
               model_name=model_name)


if __name__ == "__main__":
    main()

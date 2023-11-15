import torch
from utils import (
    prepare_vocab,
    save_model,
    parse_arguments,
    create_writer,
    plot_loss_curves,
)
from model_builder import create_GPT_model
from engine import train
from statistics import mean


def main():
    args = parse_arguments()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    block_size = 128
    max_iters = args.max_iters
    learning_rate = args.lr
    model_id = 10000

    vocab_size, encode, decode = prepare_vocab()

    model = create_GPT_model(vocab_size=vocab_size, device=device)

    model.load_state_dict(
        torch.load(
            f=f"models/GPT_Model_trained_{model_id}_epochs_medical_finetunned.pth",
            map_location=torch.device(device),
        )
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    results = train(
        model=model,
        optimizer=optimizer,
        writer=create_writer(
            experiment_name=f"{model_id}-{model_id+max_iters}_epochs",
            model_name="GPT"
        ),
        epochs=max_iters,
        encode=encode,
        device=device,
        block_size=block_size,
        batch_size=batch_size,
        finetuning=True,
        med=True,
    )

    save_model(
        model=model,
        model_name=f"GPT_Model_med_{max_iters}_epochs.pth"
    )

    plot_loss_curves(results=results)
    print(mean(results["train_loss"]))


if __name__ == "__main__":
    main()

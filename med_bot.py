import torch
from utils import prepare_vocab
from model_builder import create_GPT_model


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    vocab_size, encode, decode = prepare_vocab()

    model = create_GPT_model(vocab_size=vocab_size,
                             device=device)

    model.load_state_dict(torch.load(
        f="models/GPT_Model_med_10000_epochs.pth",
        map_location=torch.device(device)))

    inp = input("Input question: ")
    in_len = len(inp)
    prompt = torch.tensor(encode(inp),
                          dtype=torch.long, device=device)
    model.eval()
    with torch.inference_mode():
        response = model.generate(prompt.unsqueeze(0),
                                  max_new_tokens=100)[0].tolist()
    decoded_text = decode(response)
    print("A: " + decoded_text[in_len:])


if __name__ == "__main__":
    main()

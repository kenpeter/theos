"""
Chat with the trained TinyModel.
"""

import torch
import torch.nn.functional as F


def load_model(path="tiny_model.pt"):
    from tiny_model import TinyConfig, TinyModel
    cfg = TinyConfig()
    model = TinyModel(cfg)
    ckpt = torch.load(path, map_location="cpu")
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model, cfg


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="tiny_model.pt")
    parser.add_argument("--loops", type=int, default=6)
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--max_tokens", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.model)
    model = model.to(device)
    print(f"Model loaded ({model.num_parameters():,} params) on {device}")

    tokenizer = TinyTokenizer(vocab_size=cfg.vocab_size)

    print("\nChat ready. Type your messages below (Ctrl+C to exit).\n")

    while True:
        try:
            user = input("  You: ")
            if not user.strip():
                continue
            prompt = tokenizer.encode(user, max_len=cfg.max_seq_len)
            input_ids = prompt.unsqueeze(0).to(device)

            if args.max_tokens > 0:
                with torch.no_grad():
                    for _ in range(args.max_tokens):
                        logits = model(input_ids, n_loops=args.loops)
                        logits = logits[:, -1, :] / args.temp
                        if args.top_k > 0:
                            v, _ = logits.topk(args.top_k)
                            logits[logits < v[:, -1:]] = float("-inf")
                        probs = F.softmax(logits, dim=-1)
                        next_tok = torch.multinomial(probs, num_samples=1)
                        input_ids = torch.cat([input_ids, next_tok], dim=1)
                        if next_tok.item() == tokenizer.eos_id:
                            break

            reply = tokenizer.decode(input_ids[0])
            print(f"  AI: {reply}\n")
        except KeyboardInterrupt:
            print("\nBye!")
            break


if __name__ == "__main__":
    from train import TinyTokenizer
    main()
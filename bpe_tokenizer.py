"""
BPE Tokenizer — trains on dataset, encodes/decodes text & code.
"""
import os
import json
from pathlib import Path

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors


class BPETokenizer:
    SPECIAL_TOKENS = ["<pad>", "<eos>", "<unk>", "<bos>"]

    def __init__(self, vocab_size: int = 4096, path: str = "bpe_tokenizer.json"):
        self.vocab_size = vocab_size
        self.path = path
        self.pad_id = 0
        self.eos_id = 1
        self.unk_id = 2
        self.bos_id = 3

        if os.path.exists(path):
            self._tk = Tokenizer.from_file(path)
        else:
            self._tk = Tokenizer(models.BPE(unk_token="<unk>"))
            self._tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            self._tk.decoder = decoders.ByteLevel()
            self._tk.post_processor = processors.ByteLevel(trim_offsets=False)

    def train(self, texts: list[str]):
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.SPECIAL_TOKENS,
            show_progress=True,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        self._tk.train_from_iterator(texts, trainer)
        self._tk.save(self.path)
        print(f"BPE tokenizer saved to {self.path} (vocab={self._tk.get_vocab_size()})")

    def encode(self, text: str) -> list[int]:
        out = self._tk.encode(text)
        return [self.bos_id] + out.ids + [self.eos_id]

    def encode_prompt(self, text: str) -> list[int]:
        out = self._tk.encode(text)
        return out.ids

    def decode(self, ids: list[int]) -> str:
        tokens = [i for i in ids if i not in (self.pad_id, self.bos_id)]
        if self.eos_id in tokens:
            tokens = tokens[:tokens.index(self.eos_id)]
        return self._tk.decode(tokens)

    @property
    def vocab_size(self):
        return self._vocab_size

    @vocab_size.setter
    def vocab_size(self, v):
        self._vocab_size = v


if __name__ == "__main__":
    from train import load_wikitext
    texts = load_wikitext("train")[:10000]
    print(f"Training BPE on {len(texts)} texts...")
    tok = BPETokenizer(vocab_size=4096)
    tok.train(texts)

    test = "def hello():\n    print('hello world')\n    return 42"
    enc = tok.encode(test)
    dec = tok.decode(enc)
    print(f"Test encode/decode: {dec[:80]}...")
    print(f"  tokens={len(enc)} chars={len(test)} ratio={len(test)/len(enc):.2f}x")
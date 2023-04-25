import json
from pathlib import Path
from typing import Any

import torch
from transformers import BatchEncoding, PreTrainedTokenizer

from .utils.commons import intersperse
from .utils.text import _clean_text, symbols


class VitsTokenizer(PreTrainedTokenizer):
    model_input_names = ["input_ids", "input_lengths"]

    def __init__(
        self,
        vocab: list[str] = symbols,
        text_cleaners=["english_cleaners2"],
        add_blank=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.vocab = vocab
        self.text_cleaners = text_cleaners
        self.add_blank = add_blank

        self._symbol_to_id = {s: i for i, s in enumerate(vocab)}
        self._id_to_symbol = {i: s for i, s in enumerate(vocab)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> list[str]:
        return self.vocab

    def _tokenize(self, text, **kwargs) -> list[str]:
        return [symbol for symbol in text]

    def _convert_token_to_id(self, token) -> int:
        return self._symbol_to_id[token]

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_to_symbol[index]

    def prepare_for_tokenization(
        self, text: str, **kwargs
    ) -> tuple[str, dict[str, Any]]:
        return (_clean_text(text, self.text_cleaners), kwargs)

    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = None
    ) -> tuple[str]:
        path = Path(save_directory) / f"{filename_prefix or ''}vocab.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        return (str(path),)

    def prepare_for_model(self, ids, *args, **kwargs) -> BatchEncoding:
        encoded = super().prepare_for_model(
            intersperse(ids, 0) if self.add_blank else ids, *args, **kwargs
        )
        input_ids = encoded["input_ids"]
        input_lengths = (
            torch.LongTensor(input_ids.size(0))
            if isinstance(input_ids, torch.Tensor)
            else len(input_ids)  # type: ignore
        )
        encoded["input_lengths"] = input_lengths

        return encoded

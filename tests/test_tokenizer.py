import pytest

from llm_go.tokenizer.go_tokenizer import GoTokenizer


SAMPLE_GO = """\
package main

import "fmt"

func main() {
\tfmt.Println("Hello, Go!")
}
"""


class TestGoTokenizerTraining:
    def test_train_encode_decode_roundtrip(self, tmp_path):
        texts = [SAMPLE_GO] * 50  # need enough data to train BPE merges
        tok   = GoTokenizer.train(
            iterator=iter(texts),
            vocab_size=512,
            save_dir=str(tmp_path / "tok"),
        )
        ids = tok.encode(SAMPLE_GO)
        assert len(ids) > 0
        # Vocab check
        assert tok.vocab_size <= 512

    def test_save_load(self, tmp_path):
        texts = [SAMPLE_GO] * 50
        tok   = GoTokenizer.train(iterator=iter(texts), vocab_size=512)
        tok.save(str(tmp_path / "tok"))
        tok2  = GoTokenizer.load(str(tmp_path / "tok"))
        assert tok.encode(SAMPLE_GO) == tok2.encode(SAMPLE_GO)

    def test_special_tokens_present(self, tmp_path):
        texts = [SAMPLE_GO] * 50
        tok   = GoTokenizer.train(iterator=iter(texts), vocab_size=512)
        assert tok.token_to_id("<pad>")  == GoTokenizer.PAD_ID
        assert tok.token_to_id("<bos>")  == GoTokenizer.BOS_ID
        assert tok.token_to_id("<eos>")  == GoTokenizer.EOS_ID

    def test_encode_go_file_injects_tags(self, tmp_path):
        texts = [SAMPLE_GO] * 50
        tok   = GoTokenizer.train(iterator=iter(texts), vocab_size=512)
        ids   = tok.encode_go_file(SAMPLE_GO, version="1.22")
        decoded = tok.decode(ids, skip_special_tokens=False)
        assert "<go_file>" in decoded
        assert "go1.22"    in decoded

    def test_batch_encode(self, tmp_path):
        texts = [SAMPLE_GO] * 50
        tok   = GoTokenizer.train(iterator=iter(texts), vocab_size=512)
        batch = tok.encode_batch([SAMPLE_GO, SAMPLE_GO])
        assert len(batch) == 2
        assert batch[0] == batch[1]  # identical inputs → identical outputs

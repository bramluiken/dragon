# Tokenizer

This directory stores tokenizer logic and related assets.

A minimal whitespace tokenizer is implemented in `core/src/tokenizer.rs` for early experimentation. It maps space-separated words to integer ids using a provided vocabulary.

## Training a vocabulary

A simple helper CLI is available to generate a vocabulary file from a text corpus:

```bash
cargo run --bin train_vocab <input.txt> <output_vocab.txt> [limit]
```

`limit` is optional and specifies the maximum number of tokens to keep, sorted by frequency. The resulting vocabulary file contains one token per line and can be used with the CLI tools in `core/src/bin`.


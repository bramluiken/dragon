# Tokenizer

This directory stores tokenizer logic and related assets.

The project now includes a simple byte pair encoding (BPE) tokenizer implemented in `core/src/tokenizer.rs`. It loads a vocabulary and merge rules from the files in this directory.

Example files `vocab.txt` and `merges.txt` demonstrate the expected format. Each merge line lists two space-separated tokens.

## Training a vocabulary

A simple helper CLI is available to generate a vocabulary file from a text corpus:

```bash
cargo run --bin train_vocab <input.txt> <output_vocab.txt> [limit]
```

`limit` is optional and specifies the maximum number of tokens to keep, sorted by frequency. The resulting vocabulary file contains one token per line and can be used with the CLI tools in `core/src/bin`.

## Updating a vocabulary

If you already have a vocabulary and want to add tokens from additional text, use the `update_vocab` helper:

```bash
cargo run --bin update_vocab <old_vocab.txt> <new_text.txt> <output_vocab.txt> [limit]
```

New tokens are appended to the existing list sorted by frequency. When `limit` is supplied, the final vocabulary will be truncated to at most that many entries.


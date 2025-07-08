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

## Using the tokenizer

The BPE tokenizer can be instantiated directly from the vocabulary and merges
files. Below is a minimal example in Rust:

```rust
use dragon_core::tokenizer::BpeTokenizer;

let vocab = std::fs::read_to_string("data/tokenizer/vocab.txt")?
    .lines()
    .map(|s| s.to_string())
    .collect();
let merges = std::fs::read_to_string("data/tokenizer/merges.txt")?
    .lines()
    .filter_map(|l| {
        let mut p = l.split_whitespace();
        Some((p.next()?.to_string(), p.next()?.to_string()))
    })
    .collect();
let tok = BpeTokenizer::new(vocab, merges, 0);

let ids = tok.encode("hello world");
let text = tok.decode(&ids);
```

For a ready-made command-line demonstration you can run:

```bash
cargo run --bin infer_text data/tokenizer/vocab.txt data/tokenizer/merges.txt "hello world"
```

### PHP FFI example

The PHP side can call the tokenizer via FFI. The `examples/ffi_tokenize.php`
script demonstrates this:

```bash
php php/examples/ffi_tokenize.php "hello world"
```

It loads `libdragon_core.so` and prints the encoded token ids as JSON.


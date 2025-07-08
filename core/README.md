# dragon-core\n\nRust transformer core.

## CLI Example

A simple command-line tool `infer` demonstrates how to run the model on a list
of token ids:

```bash
cargo run --bin infer -- 0 1 2
```

For text-based generation you can use the `generate_text` binary:

```bash
cargo run --bin generate_text -- tokenizer/vocab.txt "hello" 3
```

To train the output layer with a short text snippet you can use `train`:

```bash
cargo run --bin train -- tokenizer/vocab.txt "hello world hello" 10
```

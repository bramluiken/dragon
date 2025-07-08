use dragon_core::model::Model;
use dragon_core::tokenizer::WhitespaceTokenizer;
use dragon_core::loss::perplexity;
use dragon_core::hyperparams::{EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_HEADS};
use std::env;
use std::fs;

fn main() {
    let mut args = env::args().skip(1);
    let vocab_path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("Usage: eval_perplexity <vocab.txt> <text>");
            std::process::exit(1);
        }
    };
    let text = match args.next() {
        Some(t) => t,
        None => {
            eprintln!("Usage: eval_perplexity <vocab.txt> <text>");
            std::process::exit(1);
        }
    };

    let vocab_contents = fs::read_to_string(vocab_path).expect("failed to read vocab file");
    let vocab: Vec<String> = vocab_contents.lines().map(|s| s.to_string()).collect();

    let tokenizer = WhitespaceTokenizer::new(vocab.clone(), 0);
    let tokens = tokenizer.encode(&text);

    if tokens.len() < 2 {
        eprintln!("Need at least two tokens to compute perplexity");
        std::process::exit(1);
    }

    let inputs = &tokens[..tokens.len() - 1];
    let targets = &tokens[1..];

    let vocab_size = vocab.len();
    let model = Model::new(vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_HEADS);
    let logits = model.forward(inputs);
    let ppl = perplexity(&logits, targets);
    println!("perplexity: {}", ppl);
}

use dragon_core::model::Model;
use dragon_core::tokenizer::BpeTokenizer;
use dragon_core::hyperparams::{EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_HEADS};
use std::env;
use std::fs;

fn main() {
    let mut args = env::args().skip(1);
    let vocab_path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("Usage: generate_text <vocab.txt> <merges.txt> <prompt> <steps>");
            std::process::exit(1);
        }
    };
    let merges_path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("Usage: generate_text <vocab.txt> <merges.txt> <prompt> <steps>");
            std::process::exit(1);
        }
    };
    let prompt = match args.next() {
        Some(t) => t,
        None => {
            eprintln!("Usage: generate_text <vocab.txt> <merges.txt> <prompt> <steps>");
            std::process::exit(1);
        }
    };
    let steps: usize = match args.next() {
        Some(s) => s.parse().expect("invalid steps"),
        None => {
            eprintln!("Usage: generate_text <vocab.txt> <merges.txt> <prompt> <steps>");
            std::process::exit(1);
        }
    };

    let vocab_contents = fs::read_to_string(vocab_path).expect("failed to read vocab file");
    let vocab: Vec<String> = vocab_contents.lines().map(|s| s.to_string()).collect();
    let merges_contents = fs::read_to_string(merges_path).expect("failed to read merges file");
    let merges: Vec<(String, String)> = merges_contents
        .lines()
        .filter_map(|l| {
            let mut parts = l.split_whitespace();
            let a = parts.next()?.to_string();
            let b = parts.next()?.to_string();
            Some((a, b))
        })
        .collect();

    let tokenizer = BpeTokenizer::new(vocab.clone(), merges, 0);
    let mut tokens = tokenizer.encode(&prompt);

    let vocab_size = vocab.len();
    let model = Model::new(vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_HEADS);
    tokens = model.generate(&tokens, steps);

    let out_text = tokenizer.decode(&tokens);
    println!("{}", out_text);
}

use dragon_core::model::Model;
use dragon_core::tokenizer::WhitespaceTokenizer;
use dragon_core::hyperparams::{EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_HEADS};
use std::env;
use std::fs;

fn main() {
    let mut args = env::args().skip(1);
    let vocab_path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("Usage: infer_text <vocab.txt> <text>");
            std::process::exit(1);
        }
    };
    let text = match args.next() {
        Some(t) => t,
        None => {
            eprintln!("Usage: infer_text <vocab.txt> <text>");
            std::process::exit(1);
        }
    };

    let vocab_contents = fs::read_to_string(vocab_path).expect("failed to read vocab file");
    let vocab: Vec<String> = vocab_contents.lines().map(|s| s.to_string()).collect();

    let tokenizer = WhitespaceTokenizer::new(vocab.clone(), 0);

    let tokens = tokenizer.encode(&text);

    let vocab_size = vocab.len();
    let model = Model::new(vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_HEADS);
    let logits = model.forward(&tokens);

    for (idx, logit) in logits.iter().enumerate() {
        println!("step {} -> {:?}", idx, logit);
    }

    let predicted: Vec<usize> = logits
        .iter()
        .map(|logit| {
            logit
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0)
        })
        .collect();
    let out_text = tokenizer.decode(&predicted);
    println!("decoded: {}", out_text);
}

use dragon_core::model::Model;
use dragon_core::hyperparams::{DEFAULT_VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_HEADS};
use std::io::{self, Write};

fn main() {
    let mut args = std::env::args().skip(1);
    let steps: usize = match args.next() {
        Some(s) => s.parse().expect("invalid steps"),
        None => {
            eprintln!("Usage: generate_tokens <steps> <token0> [token1 ...]");
            std::process::exit(1);
        }
    };

    let tokens: Vec<usize> = args.map(|a| a.parse::<usize>().expect("invalid token")).collect();
    if tokens.is_empty() {
        eprintln!("Usage: generate_tokens <steps> <token0> [token1 ...]");
        std::process::exit(1);
    }

    let mut current = tokens;
    let vocab_size = DEFAULT_VOCAB_SIZE;
    let model = Model::new(vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_HEADS);

    for _ in 0..steps {
        let logits = model.forward(&current);
        if let Some(last) = logits.last() {
            let next = last
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            current.push(next);
            println!("{}", next);
            io::stdout().flush().unwrap();
        }
    }
}
